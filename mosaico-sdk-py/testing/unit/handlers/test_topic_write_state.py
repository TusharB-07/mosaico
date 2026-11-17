"""
Unit tests for the timestamp ordering checks and the batch sizing of
`_TopicWriteState`.

The `timestamp_ns` of the records written to a topic must be non-decreasing,
within a batch and across `push_record()`/`push_batch()` calls: equal
consecutive timestamps are valid, a decreasing timestamp is rejected.

`push_batch()` sizes its batches against `max_batch_size_bytes` itself: an
oversized batch is split into chunks under the limit, smaller consecutive
batches are merged into a single message, and a single row that alone exceeds
the limit rejects the whole batch. Whatever the path, the records must reach
the server in push order, interleavings included.
"""

from unittest.mock import MagicMock

import pyarrow as pa
import pyarrow.ipc as pa_ipc
import pytest

from mosaicolabs.handlers.internal.topic_write_state import (
    _is_non_decreasing,
    _TopicWriteState,
)
from mosaicolabs.models.core import Message
from mosaicolabs.models.data import Floating64
from mosaicolabs.models.sensors import CompressedImage, ImageFormat

_DATA_SCHEMA = Floating64.__msco_pyarrow_struct__
_DATA_FIELD = _DATA_SCHEMA[0].name

_IMAGE_SCHEMA = CompressedImage.__msco_pyarrow_struct__
"""A variable-width ontology: the `data` binary column makes row sizes tunable."""


@pytest.fixture
def writer() -> MagicMock:
    return MagicMock()


@pytest.fixture
def state(writer) -> _TopicWriteState:
    return _TopicWriteState(
        topic_name="/topic",
        data_schema=_DATA_SCHEMA,
        writer=writer,
        max_batch_size_bytes=1024 * 1024,
    )


def _push_batch(state: _TopicWriteState, timestamps: list) -> None:
    state.push_batch(
        timestamp_ns=pa.array(timestamps, pa.int64()),
        data=pa.RecordBatch.from_pydict(
            {_DATA_FIELD: [float(i) for i in range(len(timestamps))]},
            schema=pa.schema(list(_DATA_SCHEMA)),
        ),
    )


def _push_image_batch(
    state: _TopicWriteState, timestamps: list, payload_size: int
) -> None:
    state.push_batch(
        timestamp_ns=pa.array(timestamps, pa.int64()),
        data=pa.RecordBatch.from_pydict(
            {
                "header": [None] * len(timestamps),
                "data": [b"x" * payload_size] * len(timestamps),
                "format": ["jpeg"] * len(timestamps),
            },
            schema=pa.schema(list(_IMAGE_SCHEMA)),
        ),
    )


def _push_record(state: _TopicWriteState, timestamp: int) -> bool:
    return state.push_record(Message(timestamp_ns=timestamp, data=Floating64(data=0.0)))


def _push_image_record(
    state: _TopicWriteState, timestamp: int, payload_size: int = 1024
) -> bool:
    return state.push_record(
        Message(
            timestamp_ns=timestamp,
            data=CompressedImage(data=b"y" * payload_size, format=ImageFormat.JPEG),
        )
    )


def _sent_timestamps(writer: MagicMock) -> list:
    """The `timestamp_ns` of every record sent on the wire, in order."""
    return [
        ts
        for call in writer.write.call_args_list
        for ts in call.args[0].column("timestamp_ns").to_pylist()
    ]


def _written_sizes(writer: MagicMock) -> list:
    """The serialized size of every batch sent on the wire, in order."""
    return [
        pa_ipc.get_record_batch_size(call.args[0])
        for call in writer.write.call_args_list
    ]


# --- _is_non_decreasing ---


@pytest.mark.parametrize(
    "values, expected",
    [
        ([], True),
        ([5], True),
        ([1, 2, 3], True),
        ([1, 1, 1], True),
        ([1, 2, 2, 3], True),
        ([2, 1], False),
        ([1, 3, 3, 2], False),
        ([1, None, 2], False),
    ],
)
def test_is_non_decreasing(values, expected):
    assert _is_non_decreasing(pa.array(values, pa.int64())) is expected


# --- Within a batch ---


@pytest.mark.parametrize("timestamps", [[10, 20, 30], [10, 10, 20], [10, 10, 10]])
def test_push_batch_accepts_non_decreasing_timestamps(state, writer, timestamps):
    _push_batch(state, timestamps)
    state.close()

    assert _sent_timestamps(writer) == timestamps
    assert state._written_records == len(timestamps)


@pytest.mark.parametrize("timestamps", [[20, 10], [10, 20, 20, 15]])
def test_push_batch_rejects_decreasing_timestamps(state, writer, timestamps):
    with pytest.raises(ValueError, match="'timestamp_ns' must be non-decreasing: "):
        _push_batch(state, timestamps)

    writer.write.assert_not_called()
    assert state._written_records == 0


# --- Across pushes ---


def test_push_record_accepts_non_decreasing_timestamps(state, writer):
    for ts in [10, 10, 20, 30, 30]:
        assert _push_record(state, ts)
    state.close()

    assert _sent_timestamps(writer) == [10, 10, 20, 30, 30]


def test_push_record_rejects_timestamp_lower_than_last(state, writer):
    _push_record(state, 20)

    with pytest.raises(ValueError, match="across pushes on topic '/topic': got 10"):
        _push_record(state, 10)

    # The rejected record is not buffered, and the writer keeps accepting records
    # that follow the last accepted one.
    assert _push_record(state, 20)
    state.close()
    assert _sent_timestamps(writer) == [20, 20]


@pytest.mark.parametrize("next_batch", [[30, 40], [40, 40]])
def test_push_batch_accepts_batch_following_previous_one(state, writer, next_batch):
    _push_batch(state, [10, 30])
    _push_batch(state, next_batch)
    state.close()

    assert _sent_timestamps(writer) == [10, 30, *next_batch]


def test_push_batch_rejects_batch_starting_before_previous_one(state, writer):
    _push_batch(state, [10, 30])

    with pytest.raises(ValueError, match="across pushes on topic '/topic': got 20"):
        _push_batch(state, [20, 40])

    state.close()
    assert _sent_timestamps(writer) == [10, 30]
    assert state._written_records == 2


def test_push_batch_rejects_batch_starting_before_buffered_record(state, writer):
    _push_record(state, 30)

    with pytest.raises(ValueError, match="across pushes"):
        _push_batch(state, [20, 40])

    # The rejected batch does not flush the buffer: the record is still pending.
    writer.write.assert_not_called()
    state.close()
    assert _sent_timestamps(writer) == [30]


def test_push_record_rejects_timestamp_lower_than_previous_batch(state, writer):
    _push_batch(state, [10, 30])

    with pytest.raises(ValueError, match="across pushes"):
        _push_record(state, 20)

    assert _push_record(state, 30)
    state.close()
    assert _sent_timestamps(writer) == [10, 30, 30]


def test_rejected_batch_does_not_move_last_timestamp(state, writer):
    _push_batch(state, [10, 20])

    # Rejected within the batch: its last timestamp (50) must not be recorded.
    with pytest.raises(ValueError):
        _push_batch(state, [40, 30, 50])

    _push_batch(state, [20, 25])
    state.close()
    assert _sent_timestamps(writer) == [10, 20, 20, 25]


def test_empty_batch_does_not_move_last_timestamp(state, writer):
    _push_batch(state, [10, 20])
    _push_batch(state, [])

    with pytest.raises(ValueError, match="across pushes"):
        _push_batch(state, [15])
    _push_batch(state, [20])


def test_oversized_record_does_not_move_last_timestamp(state, writer):
    _push_record(state, 10)

    # A record that alone exceeds the size limit is dropped, not stored: its
    # timestamp must not constrain the following records.
    max_size = state.max_batch_size_bytes
    state.max_batch_size_bytes = 1
    assert not _push_record(state, 50)
    state.max_batch_size_bytes = max_size

    assert _push_record(state, 20)
    state.close()
    assert _sent_timestamps(writer) == [10, 20]


# --- Batch sizing: split ---


def test_push_batch_splits_oversized_batch_into_chunks_under_limit(state, writer):
    timestamps = list(range(1000, 1200))
    state.max_batch_size_bytes = 1024

    _push_batch(state, timestamps)
    state.close()

    # Every message sent on the wire fits the limit, and the chunks carry all
    # the records, in order.
    assert len(writer.write.call_args_list) > 1
    assert all(size <= 1024 for size in _written_sizes(writer))
    assert _sent_timestamps(writer) == timestamps
    assert state._pushed_records == len(timestamps)
    assert state._written_records == len(timestamps)


def test_push_batch_splits_variable_width_batch_under_limit(writer):
    state = _TopicWriteState(
        topic_name="/topic",
        data_schema=_IMAGE_SCHEMA,
        writer=writer,
        max_batch_size_bytes=16 * 1024,
    )
    # Rows of alternating sizes: the merge must adapt to the actual row sizes,
    # flushing whenever the next batch would not fit
    timestamps = list(range(200))
    payload_sizes = [1024 if i % 2 == 0 else 4096 for i in range(200)]

    for ts, size in zip(timestamps, payload_sizes):
        state.push_batch(
            timestamp_ns=pa.array([ts], pa.int64()),
            data=pa.RecordBatch.from_pydict(
                {
                    "header": [None],
                    "data": [b"x" * size],
                    "format": ["jpeg"],
                },
                schema=pa.schema(list(_IMAGE_SCHEMA)),
            ),
        )
    state.close()

    assert all(size <= 16 * 1024 for size in _written_sizes(writer))
    assert _sent_timestamps(writer) == timestamps
    assert state._written_records == len(timestamps)


def test_push_batch_rejects_row_larger_than_limit(writer):
    state = _TopicWriteState(
        topic_name="/topic",
        data_schema=_IMAGE_SCHEMA,
        writer=writer,
        max_batch_size_bytes=4096,
    )

    # The middle row alone exceeds the limit: the whole batch is rejected,
    # without any partial write.
    with pytest.raises(ValueError, match="Row 1 of the batch cannot be sent"):
        state.push_batch(
            timestamp_ns=pa.array([10, 20, 30], pa.int64()),
            data=pa.RecordBatch.from_pydict(
                {
                    "header": [None] * 3,
                    "data": [b"x" * 1024, b"x" * 8192, b"x" * 1024],
                    "format": ["jpeg"] * 3,
                },
                schema=pa.schema(list(_IMAGE_SCHEMA)),
            ),
        )

    writer.write.assert_not_called()
    assert state._pushed_records == 0
    assert state._written_records == 0


def test_row_larger_than_limit_does_not_flush_message_buffer(writer):
    state = _TopicWriteState(
        topic_name="/topic",
        data_schema=_IMAGE_SCHEMA,
        writer=writer,
        max_batch_size_bytes=4096,
    )
    _push_image_record(state, 10, payload_size=1024)

    # The rejected batch must not flush the buffered record either.
    with pytest.raises(ValueError, match="cannot be sent"):
        _push_image_batch(state, [20, 30], payload_size=8192)

    writer.write.assert_not_called()

    # The state stays consistent: the buffered record is still delivered.
    state.close()
    assert _sent_timestamps(writer) == [10]


# --- Batch sizing: merge ---


def test_push_batch_merges_small_batches_into_single_message(state, writer):
    _push_batch(state, [10, 20])
    _push_batch(state, [30, 40])

    # Both batches fit together: they are merged, nothing is sent yet.
    writer.write.assert_not_called()

    state.close()

    # ...and on close they travel as a single message.
    assert len(writer.write.call_args_list) == 1
    assert _sent_timestamps(writer) == [10, 20, 30, 40]
    assert state._written_records == 4


def test_push_batch_flushes_merge_buffer_when_next_batch_exceeds_limit(state, writer):
    # One two-row batch fits exactly: two of them don't.
    wire_batch = pa.RecordBatch.from_arrays(
        [pa.array([10, 20], pa.int64()), pa.array([0.0, 0.0])],
        schema=state.wire_schema,
    )
    state.max_batch_size_bytes = pa_ipc.get_record_batch_size(wire_batch)

    _push_batch(state, [10, 20])
    assert writer.write.call_count == 0

    _push_batch(state, [30, 40])
    # The first batch was flushed to make room for the second one.
    assert writer.write.call_count == 1
    assert _sent_timestamps(writer) == [10, 20]

    state.close()
    assert writer.write.call_count == 2
    assert _sent_timestamps(writer) == [10, 20, 30, 40]


def test_merge_never_exceeds_limit(state, writer):
    # Many small batches, flushed only when the next one would not fit.
    state.max_batch_size_bytes = 2048
    timestamps = list(range(100))
    for ts in timestamps:
        _push_batch(state, [ts])

    state.close()

    assert all(size <= 2048 for size in _written_sizes(writer))
    assert _sent_timestamps(writer) == timestamps
    assert state._written_records == len(timestamps)


# --- Batch sizing: ordering across the two buffers ---


def test_merged_batches_are_flushed_before_pushed_records(state, writer):
    # A buffered batch must never be overtaken by messages pushed after it.
    _push_batch(state, [10, 20])
    _push_record(state, 30)
    _push_batch(state, [40, 50])

    # The second push_batch flushes the buffered record, which flushes the
    # older merged batch first.
    assert writer.write.call_count == 2
    assert _sent_timestamps(writer) == [10, 20, 30]

    state.close()
    assert _sent_timestamps(writer) == [10, 20, 30, 40, 50]


def test_split_chunks_follow_buffered_batches_and_messages(writer):
    state = _TopicWriteState(
        topic_name="/topic",
        data_schema=_IMAGE_SCHEMA,
        writer=writer,
        max_batch_size_bytes=8192,
    )
    _push_image_batch(state, [10, 20], payload_size=1024)
    _push_image_record(state, 30, payload_size=1024)

    # An oversized batch, split into chunks, must arrive after both.
    _push_image_batch(state, list(range(40, 60)), payload_size=2048)
    state.close()

    assert _sent_timestamps(writer) == [10, 20, 30, *range(40, 60)]
    assert all(size <= 8192 for size in _written_sizes(writer))
    assert state._written_records == 23
