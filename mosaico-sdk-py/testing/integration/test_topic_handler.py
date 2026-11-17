"""
Integration tests for the lazy fetch of a topic's PyArrow ontology schema.

`TopicHandler` defers fetching the topic's Arrow schema to the first access of
`ontology_schema`: a `get_schema` call makes the server read the topic data
from the store, so neither connecting the handler nor reading its metadata
must trigger it. Once fetched, the schema is cached.

The `get_schema` calls are observed through a transparent wrapper around the
real FlightClient of the connection: every response comes from the live
server, no SDK internal is replaced. These tests used to live in the unit
suite (`testing/unit/handlers/test_topic_handler.py`) as fully mocked tests.
"""

from dataclasses import replace
from typing import List

import pytest
from pyarrow.flight import FlightClient

from mosaicolabs.comm import MosaicoClient
from mosaicolabs.helpers import pack_topic_resource_name
from mosaicolabs.models.core import Message
from mosaicolabs.models.data import Floating64
from mosaicolabs.proto.v1 import flight_pb2

_SEQUENCE_NAME = "test-topic-handler-schema-sequence"
_TOPIC_NAME = "/floating"
_TIMESTAMPS_NS = [1700000000000000000, 1700000000000000123]


class _SchemaFetchSpy:
    """
    Transparent wrapper around the real `FlightClient` of a connection, which
    records the descriptor of every `get_schema` call.

    Everything is delegated to the wrapped client, so the server stays in the
    loop: the wrapper only makes the schema fetches observable.
    """

    def __init__(self, flight_client: FlightClient):
        self._flight_client = flight_client
        self.schema_descriptors: List = []

    def get_schema(self, descriptor):
        self.schema_descriptors.append(descriptor)
        return self._flight_client.get_schema(descriptor)

    def __getattr__(self, name):
        return getattr(self._flight_client, name)


@pytest.fixture
def schema_spy(mosaico_client: MosaicoClient) -> _SchemaFetchSpy:
    """Wraps the FlightClient of the client connection, recording schema fetches."""
    connection = mosaico_client._connection
    spy = _SchemaFetchSpy(connection.flight_client)
    # `ConnectionContext` is a frozen dataclass, so the spy cannot be assigned
    # to its field: the client is handed an equivalent context carrying the
    # spy instead. Every handler and writer created afterwards receives it,
    # and every call but `get_schema` is delegated to the wrapped client.
    mosaico_client._connection = replace(connection, flight_client=spy)
    return spy


@pytest.fixture
def floating_topic(mosaico_client: MosaicoClient, schema_spy):
    """A finalized sequence holding a `Floating64` topic with two records."""
    with mosaico_client.sequence_create(_SEQUENCE_NAME, {}) as swriter:
        twriter = swriter.topic_create(_TOPIC_NAME, {}, Floating64)
        assert twriter is not None
        for timestamp_ns in _TIMESTAMPS_NS:
            twriter.push(Message(timestamp_ns=timestamp_ns, data=Floating64(data=0.5)))

    yield

    mosaico_client.sequence_delete(_SEQUENCE_NAME)


def _connect_handler(
    mosaico_client: MosaicoClient,
):
    """Connects a `TopicHandler` for the floating topic, failing if not found."""
    handler = mosaico_client.topic_handler(_SEQUENCE_NAME, _TOPIC_NAME)
    assert handler is not None
    return handler


def test_connecting_a_topic_handler_does_not_fetch_the_schema(
    mosaico_client: MosaicoClient, floating_topic, schema_spy
):
    schema_spy.schema_descriptors.clear()

    handler = _connect_handler(mosaico_client)

    # The metadata comes from the flight info: no schema fetch, no data read.
    assert schema_spy.schema_descriptors == []
    assert handler.name == _TOPIC_NAME
    assert handler.sequence_name == _SEQUENCE_NAME
    assert handler.timestamp_ns_min == _TIMESTAMPS_NS[0]
    assert handler.timestamp_ns_max == _TIMESTAMPS_NS[-1]


def test_metadata_access_does_not_fetch_the_schema(
    mosaico_client: MosaicoClient, floating_topic, schema_spy
):
    handler = _connect_handler(mosaico_client)
    schema_spy.schema_descriptors.clear()

    # Reading the metadata must not read the topic data from the store.
    handler.name
    handler.sequence_name
    handler.user_metadata
    handler.created_timestamp
    handler.ontology_tag
    handler.serialization_format
    handler.total_size_bytes
    handler.chunks_number
    handler.timestamp_ns_min
    handler.timestamp_ns_max

    assert schema_spy.schema_descriptors == []


def test_ontology_schema_is_fetched_lazily_and_cached(
    mosaico_client: MosaicoClient, floating_topic, schema_spy
):
    handler = _connect_handler(mosaico_client)
    schema_spy.schema_descriptors.clear()

    # First access: fetched from the server...
    assert handler.ontology_schema == Floating64.__msco_pyarrow_struct__
    # ...second access: served from the local cache, no further fetch.
    assert handler.ontology_schema == Floating64.__msco_pyarrow_struct__

    assert len(schema_spy.schema_descriptors) == 1
    descriptor = schema_spy.schema_descriptors[0]
    cmd = flight_pb2.GetSchemaCmd.FromString(descriptor.command)
    assert cmd.locator == pack_topic_resource_name(_SEQUENCE_NAME, _TOPIC_NAME)
