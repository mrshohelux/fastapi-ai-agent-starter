from app.rag import TfidfRetriever


def test_empty_index_returns_nothing():
    r = TfidfRetriever()
    r.index({})
    assert r.retrieve("anything") == []


def test_retrieves_most_relevant_doc():
    r = TfidfRetriever()
    r.index(
        {
            "shipping": "Orders ship within 2 business days via courier.",
            "returns": "Returns are accepted within 30 days with a receipt.",
            "hours": "Our support team is available 9am to 6pm on weekdays.",
        }
    )
    results = r.retrieve("how long do I have to return an item?", top_k=1)
    assert len(results) == 1
    assert results[0].doc_id == "returns"
    assert results[0].score > 0


def test_top_k_is_respected():
    r = TfidfRetriever()
    r.index(
        {
            "a": "automation workflow integration api",
            "b": "automation workflow api webhook",
            "c": "completely unrelated topic about gardening",
        }
    )
    results = r.retrieve("automation api workflow", top_k=2)
    assert len(results) == 2
    assert {d.doc_id for d in results} == {"a", "b"}


def test_no_match_returns_empty():
    r = TfidfRetriever()
    r.index({"a": "gardening and plants"})
    results = r.retrieve("quantum computing hardware", top_k=3)
    assert results == []
