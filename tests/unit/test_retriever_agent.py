from __future__ import annotations

from types import SimpleNamespace

from src.agents.retriever_agent import RetrieverAgent


class _FakeHybridSearch:
    def __init__(self, results):
        self._results = results

    def search(self, **kwargs):
        return self._results


class _FakeReranker:
    is_enabled = False


def test_retriever_agent_falls_back_to_raw_results_for_ppt_images():
    relevant_candidate = SimpleNamespace(
        text="严格相关正文",
        metadata={"doc_hash": "a", "page_num": 1, "source_path": "doc-a.pdf"},
        score=0.8,
    )
    raw_only_visual = SimpleNamespace(
        text="含图片的视觉片段",
        metadata={"doc_hash": "b", "page_num": 2, "source_path": "doc-b.pdf", "images": [{"id": "img_1", "path": "x"}]},
        score=0.6,
    )

    calls = []

    def fake_extract(results, **kwargs):
        calls.append({
            "count": len(results),
            "topic": kwargs.get("topic"),
            "doc_hashes": [str((item.metadata or {}).get("doc_hash")) for item in results],
        })
        if kwargs.get("topic"):
            return []
        return [{"image_id": "img_1", "url": "/lesson-plan-image/img_1"}]

    agent = RetrieverAgent(
        hybrid_search=_FakeHybridSearch([relevant_candidate, raw_only_visual]),
        reranker=_FakeReranker(),
        trace=SimpleNamespace(),
        top_k=5,
        prioritize_visual_results=lambda results, query_plan: results,
        relevance_check=lambda topic, result: result is relevant_candidate,
        extract_image_resources=fake_extract,
        sanitize_source_path=lambda value: str(value),
        image_storage=None,
        collection="default",
        template_category="ppt",
        enable_rerank=False,
        enable_image_extraction=True,
    )

    message = SimpleNamespace(
        artifacts={"query_plan": {"user_query": "牛顿第三定律"}},
        context={"topic": "牛顿第三定律"},
        next_action=None,
    )

    result = agent.run(message)

    assert len(calls) == 2
    assert calls[0]["topic"] == "牛顿第三定律"
    assert calls[0]["doc_hashes"] == ["a"]
    assert calls[1]["topic"] is None
    assert calls[1]["doc_hashes"] == ["a", "b"]
    assert result.artifacts["image_resources"] == [{"image_id": "img_1", "url": "/lesson-plan-image/img_1"}]
