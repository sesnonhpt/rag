from types import SimpleNamespace

from src.libs.vector_store.qdrant_store import QdrantStore, UnexpectedResponse


class _FakeClient:
    def __init__(self):
        self.scroll_calls = []

    def retrieve(self, **kwargs):
        raise UnexpectedResponse.for_response(
            response=SimpleNamespace(
                status_code=400,
                content=b"bad id",
                reason_phrase="Bad Request",
                headers={},
            ),
        )

    def scroll(self, **kwargs):
        self.scroll_calls.append(kwargs)
        chunk_id = kwargs["scroll_filter"].must[0].match.value
        point = SimpleNamespace(
            id="2ff9e7dc-70ce-4b2f-98f2-111111111111",
            payload={
                "text": f"text for {chunk_id}",
                "chunk_id": chunk_id,
                "source_path": "demo.pdf",
            },
        )
        return [point], None


def test_get_by_ids_falls_back_to_payload_chunk_id_lookup():
    store = QdrantStore.__new__(QdrantStore)
    store.collection_name = "default"
    store._client = _FakeClient()

    records = store.get_by_ids(["legacy_chunk_001"])

    assert records == [
        {
            "id": "2ff9e7dc-70ce-4b2f-98f2-111111111111",
            "text": "text for legacy_chunk_001",
            "metadata": {
                "chunk_id": "legacy_chunk_001",
                "source_path": "demo.pdf",
            },
        }
    ]
    assert len(store._client.scroll_calls) == 1
