from garuda_edge.llm_analyst.ollama_client import OllamaClient


def test_ollama_unavailable_returns_safe_fallback(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("down")

    monkeypatch.setattr("requests.post", boom)
    result = OllamaClient().generate_json("prompt", timeout=1)
    assert result["human_review_required"] is True
    assert result["decision"] == "notify_operator"
