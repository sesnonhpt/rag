from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.schemas.api_models import ChatRequest
from app.services.chat_service import generate_chat_response
from fastapi import HTTPException


class _DummyResult:
    def __init__(self, *, text: str, score: float = 0.9, source_path: str = "/data/pdf/a.pdf"):
        self.text = text
        self.score = score
        self.metadata = {"source_path": source_path}


class _DummyHybridSearch:
    def __init__(self, results):
        self._results = results
        self.calls = []

    def search(self, *, query, top_k, filters, trace):
        self.calls.append({"query": query, "top_k": top_k, "filters": filters, "trace": trace})
        return self._results


class _DummyReranker:
    def __init__(self, *, enabled: bool = True):
        self.is_enabled = enabled
        self.calls = []

    def rerank(self, *, query, results, top_k, trace):
        self.calls.append({"query": query, "results": results, "top_k": top_k, "trace": trace})
        return SimpleNamespace(results=results)


class _DummyLLM:
    def __init__(self, content: str = "ok"):
        self._content = content
        self.calls = []

    def chat(self, messages):
        self.calls.append(messages)
        return SimpleNamespace(content=self._content)


def _make_request(*, hybrid_search, reranker, llm, fusion_top_k: int = 7):
    state = SimpleNamespace(
        settings=SimpleNamespace(retrieval=SimpleNamespace(fusion_top_k=fusion_top_k)),
        hybrid_search=hybrid_search,
        reranker=reranker,
        llm=llm,
    )
    return SimpleNamespace(app=SimpleNamespace(state=state))


def test_generate_chat_response_success_path():
    req = ChatRequest(question="牛顿第一定律是什么？", use_rerank=True)
    hybrid = _DummyHybridSearch([_DummyResult(text="牛顿第一定律内容")])
    reranker = _DummyReranker(enabled=True)
    llm = _DummyLLM(content="这是答案")
    request = _make_request(hybrid_search=hybrid, reranker=reranker, llm=llm, fusion_top_k=5)

    response = generate_chat_response(req, request)

    assert response.answer == "这是答案"
    assert len(response.citations) == 1
    assert hybrid.calls[0]["top_k"] == 5
    assert len(reranker.calls) == 1
    assert len(llm.calls) == 1


def test_generate_chat_response_retrieval_error():
    class _BadHybridSearch:
        def search(self, **kwargs):
            raise RuntimeError("boom")

    req = ChatRequest(question="test")
    request = _make_request(
        hybrid_search=_BadHybridSearch(),
        reranker=_DummyReranker(enabled=False),
        llm=_DummyLLM(),
    )

    with pytest.raises(HTTPException) as exc:
        generate_chat_response(req, request)

    assert exc.value.status_code == 500
    assert exc.value.detail["code"] == "RETRIEVAL_ERROR"


def test_generate_chat_response_no_results_uses_fallback_prompt():
    req = ChatRequest(question="不存在的问题")
    hybrid = _DummyHybridSearch([])
    reranker = _DummyReranker(enabled=True)
    llm = _DummyLLM(content="基于自身知识回答")
    request = _make_request(hybrid_search=hybrid, reranker=reranker, llm=llm)

    response = generate_chat_response(req, request)

    assert response.answer == "基于自身知识回答"
    assert response.citations == []
    assert len(reranker.calls) == 0
    messages = llm.calls[0]
    assert len(messages) == 2
    assert "当前知识库中没有找到" in messages[0].content
