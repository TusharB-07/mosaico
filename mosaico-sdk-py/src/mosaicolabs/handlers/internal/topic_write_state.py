"""
Internal Write State Module.

This module handles the low-level data buffering, serialization, and transmission
logic for a single topic. It implements the remote transmission pipeline,
while optimizing the throughput and preventing memory exhaustion.
"""

import time
from collections import defaultdict
from typing import List, Optional

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.flight as fl
import pyarrow.ipc as pa_ipc

from mosaicolabs.logging_config import get_logger
from mosaicolabs.models.core import Message

# Set the hierarchical logger
logger = get_logger(__name__)


def _encode_messages(objs: list[Message]):
    """Helper to pivot a list of Message objects into a columnar dictionary."""
    result = defaultdict(list)
    for obj in objs:
        for k, v in obj._encode().items():
            result[k].append(v)
    return dict(result)


def _is_non_decreasing(arr: pa.Array) -> bool:
    """Checks that an array is non-decreasing (equal consecutive values are allowed)"""

    # If there is less than 2 elements, it is non-decreasing
    if len(arr) < 2:
        return True

    # If contains null cannot be non-decreasing
    if arr.null_count:  # nulls would otherwise be silently skipped
        return False

    left, right = arr[:-1], arr[1:]
    return pc.all(pc.less_equal(left, right)).as_py()


class _TopicWriteState:
    """
    Manages the write buffer and async dispatch for a single topic.

    **Architecture:**
    1.  **Buffering**: Accumulates `Message` objects in `_current_data_batch`,
        flushing whenever adding the next record would exceed `max_batch_size_bytes`.
        A single record that alone exceeds `max_batch_size_bytes` cannot be split
        any further, so it is dropped instead - see `_push_by_bytes_size()`.
        `push_batch()` batches are accumulated in a separate Arrow-level buffer
        (`_pending_arrow_batches`), merged into chunks as large as the limit allows:
        they are concatenated on flush, so consecutive small batches travel as a
        single Flight message instead of one message each.
    2.  **Splitting**: a `push_batch()` batch whose serialized size exceeds
        `max_batch_size_bytes` is sliced into chunks under the limit before being
        written (see `_split_batch()`).
    3.  **Sync Dispatch**.

    The two buffers are flushed in push order: the Arrow-level buffer only holds
    batches pushed after every buffered `Message`, so it is always flushed first
    (see `_write_current_batch()`).
    """

    def __init__(
        self,
        topic_name: str,
        data_schema: pa.StructType,
        writer: Optional[fl.FlightStreamWriter],
        max_batch_size_bytes: int,
    ):
        """
        Initializes the write state.

        Args:
            topic_name (str): Topic name.
            data_schema (pa.StructType): The pyarrow schema of the ontology payload
                writing to this topic, used to build the combined message+payload
                schema for each flushed `RecordBatch`.
            writer (Optional[fl.FlightStreamWriter]): Active Flight stream writer.
            max_batch_size_bytes (Optional[int]): flush threshold for byte mode.
        """
        if writer is None:
            raise ValueError("Cannot initialize _TopicState: 'writer' is None.")

        self.topic_name: str = topic_name
        self.writer: Optional[fl.FlightStreamWriter] = writer
        self.data_schema: pa.StructType = data_schema
        self.max_batch_size_bytes = max_batch_size_bytes
        # The schema of the record batches sent on the wire (envelope + payload fields)
        self.wire_schema: pa.Schema = Message._get_schema(data_schema)

        # --- Buffering State ---
        self._current_data_batch: List[Message] = []
        self._current_batch_size_bytes: int = 0

        # --- Arrow-Level Buffering State (push_batch) ---
        # Record batches, in wire format, buffered by `push_batch()` and merged
        # into a single message on flush. Invariant: whenever the `Message` buffer
        # is not empty, these batches are older than every buffered `Message`, so
        # flushing them first (see `_write_current_batch()`) keeps push order.
        self._pending_arrow_batches: List[pa.RecordBatch] = []
        self._pending_arrow_batches_size: int = 0

        self._written_records = 0
        self._pushed_records = 0
        self._oversized_records = 0

        # --- Ordering State ---
        # The highest `timestamp_ns` accepted so far, by either `push_record()` or
        # `push_batch()`. Since accepted timestamps are non-decreasing, it is also
        # the last one. `None` until the first record is accepted.
        self._last_timestamp_ns: Optional[int] = None

    def _check_follows_last_timestamp(self, first_timestamp_ns: int) -> None:
        """
        Ensures the topic's timestamps stay non-decreasing across pushes.

        Args:
            first_timestamp_ns (int): The lowest timestamp of the records about to be
                accepted (the first one, as they are non-decreasing among themselves).

        Raises:
            ValueError: If `first_timestamp_ns` is lower than the last accepted timestamp.
        """
        if (
            self._last_timestamp_ns is not None
            and first_timestamp_ns < self._last_timestamp_ns
        ):
            raise ValueError(
                f"'timestamp_ns' must be non-decreasing across pushes on topic "
                f"'{self.topic_name}': got {first_timestamp_ns}, lower than the last "
                f"pushed timestamp {self._last_timestamp_ns}."
            )

    def _get_record_batch(self, msgs: List[Message]) -> pa.RecordBatch:
        """
        [CPU Bound] Converts Python objects to Arrow RecordBatch.
        """
        if self.writer is None:
            raise ValueError("Writer is None")

        return pa.RecordBatch.from_pydict(
            _encode_messages(msgs),
            schema=self.wire_schema,
        )

    def _get_serialized_size(self, batch: pa.RecordBatch) -> int:
        """
        Calculates exact serialized size of a RecordBatch in Arrow IPC format.

        Uses PyArrow's native C++ implementation for optimal performance.
        This method is significantly faster than full serialization and provides
        accurate size for Flight transmission limits.

        Note: Returns batch size only (excludes schema message overhead).
        """
        return pa_ipc.get_record_batch_size(batch)

    def _push_by_bytes_size(self, msg: Message) -> bool:
        """
        Buffer logic for Byte-Mode topics (e.g., Images).

        1. Serializes the *single* new record to check its size.
        2. If adding it exceeds `max_batch_size_bytes`, flushes current buffer.
        3. Adds record to new buffer.
        """
        assert self.writer is not None

        # Measure size of the new message
        single_record_batch = self._get_record_batch([msg])
        single_record_size = self._get_serialized_size(single_record_batch)

        # The record cannot be split any further: it is dropped, and the caller
        # (TopicWriter.push()) is responsible for surfacing this to the user
        # (status/last_error) and reporting it to the server as a notification.
        if single_record_size > self.max_batch_size_bytes:
            logger.error(
                f"Single record size ({single_record_size} bytes) exceeds gRPC limit "
                f"({self.max_batch_size_bytes} bytes) for topic '{self.topic_name}'. "
                "Record will be skipped."
            )
            return False

        # Check Buffer Threshold
        projected_size = self._current_batch_size_bytes + single_record_size

        if projected_size > self.max_batch_size_bytes:
            # Flush existing data, without the last message
            if self._current_data_batch:
                self._write_current_batch()
            # Now reset the buffer and add the last message
            self._current_data_batch = [msg]
            self._current_batch_size_bytes = single_record_size
        else:
            # Keep appending until limit is reached
            self._current_data_batch.append(msg)
            self._current_batch_size_bytes += single_record_size

        return True

    def push_record(self, msg: Message) -> bool:
        """
        Adds a record to the buffer.

        Automatically delegates to `_push_by_bytes_size`
        based on the ontology type defined in the message.

        Args:
            msg (Message): The message to push in the buffer for later sending.

        Returns:
            bool: True if the record was buffered for transmission. False if it
                was rejected because, on its own, it exceeds `max_batch_size_bytes`
                and therefore cannot be split any further. A rejected record does
                not take part in the ordering check of the following records.

        Raises:
            ValueError: If the writer is None, or if `msg.timestamp_ns` is lower than
                the last timestamp accepted on this topic.
        """
        if self.writer is None:
            raise ValueError("write() called on uninitialized state.")

        self._check_follows_last_timestamp(msg.timestamp_ns)

        pushed = self._push_by_bytes_size(msg)

        if pushed:
            self._pushed_records += 1
            self._last_timestamp_ns = msg.timestamp_ns
        else:
            self._oversized_records += 1

        return pushed

    def _split_batch(self, batch: pa.RecordBatch) -> List[pa.RecordBatch]:
        """
        Slices an oversized `push_batch()` batch into chunks under the limit.

        The largest prefix of rows that fits `max_batch_size_bytes` is found with a
        binary search on the serialized size of the (zero-copy) row slices, then
        repeated on the remaining rows: the serialized size of a row prefix is
        non-decreasing in the number of rows, so the search is exact.

        Args:
            batch (pa.RecordBatch): The wire-format batch to slice, whose serialized
                size exceeds `max_batch_size_bytes`.

        Returns:
            List[pa.RecordBatch]: The chunks, each under the limit, whose
                concatenation is `batch`.

        Raises:
            ValueError: If a single row alone exceeds `max_batch_size_bytes`: it
                cannot be split any further, so the whole batch is rejected.
        """
        chunks: List[pa.RecordBatch] = []
        offset = 0
        while offset < batch.num_rows:
            remaining = batch.slice(offset)

            if (
                self._get_serialized_size(remaining.slice(0, 1))
                > self.max_batch_size_bytes
            ):
                raise ValueError(
                    f"Row {offset} of the batch cannot be sent: its size alone "
                    f"exceeds the maximum batch size ({self.max_batch_size_bytes} "
                    f"bytes) for topic '{self.topic_name}' and it cannot be split "
                    "further."
                )

            # Largest prefix of the remaining rows that fits the limit
            lo, hi = 1, remaining.num_rows
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if (
                    self._get_serialized_size(remaining.slice(0, mid))
                    <= self.max_batch_size_bytes
                ):
                    lo = mid
                else:
                    hi = mid - 1

            chunks.append(remaining.slice(0, lo))
            offset += lo

        return chunks

    def _write_arrow_batch(self, batch: pa.RecordBatch) -> None:
        """
        Writes a single wire-format batch to the Flight stream.

        Args:
            batch (pa.RecordBatch): The batch to write, already under
                `max_batch_size_bytes`.

        Raises:
            Exception: If the transmission fails.
        """
        assert self.writer is not None

        try:
            self.writer.write(batch)
        except Exception as e:
            raise Exception(f"Write failed for topic '{self.topic_name}': '{e}'")

        self._written_records += batch.num_rows

    def _flush_arrow_batches(self) -> None:
        """
        Flushes the Arrow-level buffer of `push_batch()`, merging its batches.

        The buffered batches are concatenated into a single wire-format batch
        (their total size never exceeds `max_batch_size_bytes`, by construction),
        so consecutive small batches are stored as a single chunk instead of one
        chunk each.
        """
        assert self.writer is not None

        if not self._pending_arrow_batches:
            return

        batches = self._pending_arrow_batches
        self._pending_arrow_batches = []
        self._pending_arrow_batches_size = 0

        batch = batches[0] if len(batches) == 1 else pa.concat_batches(batches)
        self._write_arrow_batch(batch)

    def push_batch(self, timestamp_ns: pa.Array, data: pa.RecordBatch):
        """
        Buffers a whole batch of records, bypassing the `Message` buffer.

        The `timestamp_ns` envelope column is prepended to the payload columns,
        producing a batch with the same wire schema used for buffered messages
        (see `Message._get_schema`). The batch is then sized against
        `max_batch_size_bytes`, transparently for the caller:

        - An oversized batch is **split** into chunks under the limit, which are
          written one after the other.
        - A batch that fits is **buffered** and merged with the batches already
          buffered by the previous `push_batch()` calls: on flush they are
          concatenated into a single message, as large as the limit allows. The
          buffer is flushed before any `push()` flush (so records keep reaching
          the server in push order) and on close.

        Args:
            timestamp_ns (pa.Array): The `int64` timestamps in nanoseconds of the records,
                one per row of `data`.
            data (pa.RecordBatch): The payload columns, matching the topic's ontology schema.

        Raises:
            ValueError: If the writer is None, if `data` does not match the ontology schema,
                if `timestamp_ns` is not a null-free, non-decreasing `int64` array as long
                as `data`, if its first value is lower than the last timestamp accepted on
                this topic, or if a single row of the batch exceeds `max_batch_size_bytes`
                (it cannot be split any further).
        """
        if self.writer is None:
            raise ValueError("push_batch() called on uninitialized state.")

        expected_data_schema = pa.schema(list(self.data_schema))
        if not data.schema.equals(expected_data_schema, check_metadata=False):
            raise ValueError(
                f"Batch schema does not match the ontology schema of topic '{self.topic_name}'.\n"
                f"Expected:\n{expected_data_schema}\nGot:\n{data.schema}"
            )

        if timestamp_ns.type != pa.int64():
            raise ValueError(
                f"'timestamp_ns' must be of type int64, got '{timestamp_ns.type}'."
            )
        if timestamp_ns.null_count > 0:
            raise ValueError("'timestamp_ns' must not contain null values.")
        if len(timestamp_ns) != data.num_rows:
            raise ValueError(
                f"'timestamp_ns' has {len(timestamp_ns)} values, but the batch has "
                f"{data.num_rows} rows."
            )
        if not _is_non_decreasing(timestamp_ns):
            raise ValueError(
                "'timestamp_ns' must be non-decreasing: found a timestamp lower "
                "than the previous one."
            )
        if len(timestamp_ns) > 0:
            self._check_follows_last_timestamp(timestamp_ns[0].as_py())

        batch = pa.RecordBatch.from_arrays(
            [timestamp_ns, *data.columns], schema=self.wire_schema
        )
        batch_size = self._get_serialized_size(batch)

        # Split an oversized batch before touching any buffer: a row that alone
        # exceeds the limit rejects the whole batch without side effects.
        chunks: Optional[List[pa.RecordBatch]] = None
        if batch_size > self.max_batch_size_bytes:
            chunks = self._split_batch(batch)

        # Keep the records ordered with respect to the buffered messages: when
        # the `Message` buffer is flushed, the (older) Arrow-level buffer is
        # flushed with it.
        if self._current_data_batch:
            self._write_current_batch()

        if chunks is not None:
            # The buffered batches are older than the chunks: flush them first
            self._flush_arrow_batches()
            for chunk in chunks:
                self._write_arrow_batch(chunk)
        else:
            # Merge: flush the buffered batches if adding this one would exceed
            # the limit, then buffer it for a future merged flush
            if (
                self._pending_arrow_batches_size + batch_size
                > self.max_batch_size_bytes
            ):
                self._flush_arrow_batches()
            self._pending_arrow_batches.append(batch)
            self._pending_arrow_batches_size += batch_size

        self._pushed_records += batch.num_rows
        if batch.num_rows > 0:
            self._last_timestamp_ns = timestamp_ns[-1].as_py()

    def _submit_write_task(self, msgs_to_write: List[Message]):
        """
        Dispatches the write operation.
        """
        if self.writer is None:
            logger.error(
                f"Cannot write batch for topic '{self.topic_name}'. Writer is None."
            )
            return

        try:
            # Serialization (CPU)
            batch = self._get_record_batch(msgs_to_write)
            # Transmission (IO)
            tstart = time.time()
            self.writer.write(batch)
            logger.debug(
                f"'writer.write' MB {self._get_serialized_size(batch) / (1024 * 1024)}, time: {time.time() - tstart}"
            )
            self._written_records += len(msgs_to_write)
        except Exception as e:
            raise Exception(f"Write failed for topic '{self.topic_name}': '{e}'")

    def _write_current_batch(self):
        """
        Flushes buffer: synchronously writes the buffered records and resets buffer.
        """
        if self.writer is None:
            raise ValueError("Writer is None")

        if self._current_data_batch:
            # The Arrow-level buffer holds `push_batch()` batches that were pushed
            # before these messages: flush it first, so the records reach the
            # server in push order.
            self._flush_arrow_batches()

            records = self._current_data_batch

            # Reset immediately
            self._current_data_batch = []
            self._current_batch_size_bytes = 0

            self._submit_write_task(records)

    def close(self, with_error: bool = False):
        """
        Finalizes the topic stream.

        1. Flushes remaining buffer (unless error).
        2. Waits for pending tasks.
        3. Closes Flight writer.
        """
        if self.writer is not None:
            try:
                if not with_error:
                    # Flush any data remaining in the buffers: the Arrow-level
                    # batches first (they are older), then the `Message` buffer
                    # (`_write_current_batch()` flushes both, in order)
                    self._write_current_batch()
                    self._flush_arrow_batches()

                self.writer.done_writing()
                logger.info(
                    f"Topic '{self.topic_name}' finished. "
                    f"Pushed: {self._pushed_records}, Written: {self._written_records}, "
                    f"Dropped (oversized): {self._oversized_records}"
                )
            finally:
                self.writer.close()
                self.writer = None
