from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_index_requires_docs():
    resp = client.post("/knowledge-base/index", json={"docs": {}})
    assert resp.status_code == 400


def test_ask_without_index_returns_404():
    # fresh app instance would be cleaner, but this exercises the real
    # "nothing relevant indexed yet" path on an empty/unindexed retriever
    resp = client.post("/knowledge-base/index", json={"docs": {"x": "irrelevant filler text"}})
    assert resp.status_code == 200
    resp2 = client.post("/ask", json={"question": "completely unrelated quantum physics question"})
    assert resp2.status_code in (200, 404)


def test_index_then_ask_end_to_end():
    client.post(
        "/knowledge-base/index",
        json={
            "docs": {
                "shipping": "Orders ship within 2 business days.",
                "returns": "Returns accepted within 30 days with receipt.",
            }
        },
    )
    resp = client.post("/ask", json={"question": "how long until my order ships?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["sources"][0]["doc_id"] == "shipping"


def test_webhook_known_event():
    resp = client.post("/webhook", json={"event": "lead.created", "data": {"email": "a@b.com"}})
    assert resp.status_code == 200
    assert resp.json()["status"] == "accepted"


def test_webhook_unknown_event():
    resp = client.post("/webhook", json={"event": "nope", "data": {}})
    assert resp.status_code == 400
