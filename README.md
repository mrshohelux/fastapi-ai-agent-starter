# FastAPI AI-Agent Starter

A small, fully working FastAPI service showing the pattern I use as the "brain" behind client automations at [ShopHelp](https://shophelp.chat) and [NeverBlast](https://neverblast.com): a webhook receiver for inbound events, and a retrieval-augmented `/ask` endpoint that grounds answers in an indexed knowledge base instead of letting an LLM freewheel.

**Runs with zero API keys.** The retrieval layer defaults to a dependency-free TF-IDF retriever (`scikit-learn`) so the whole thing — including the test suite — works offline. The generation step (`compose_answer` in `app/main.py`) is the one function meant to be swapped for a real chat-completion call in a private deployment; it's left as an explicit stub rather than hidden behind a fake abstraction.

## Why it's built this way

- **Retrieval is behind an interface (`Retriever` protocol in `app/rag.py`), not hardcoded.** Start with TF-IDF (free, fast, good enough for small knowledge bases), upgrade to real embeddings (`OpenAIRetriever`, sketched but not wired to a key) only once lexical matching stops being accurate enough. That's a real cost/complexity tradeoff I make in client work, not a toy abstraction.
- **A lightweight suffix-stemmer** (`_lightly_stem` in `app/rag.py`) so "returns"/"returning"/"shipped"/"shipping" match without pulling in a full NLP dependency — picked after actually testing the naive version and watching it fail on "how long until my order ships?" vs. a doc that says "ship".
- **The webhook endpoint mirrors what n8n/Zapier/Make send**, because that's the real integration point — this is meant to sit behind an automation graph, not replace one.

## Run it

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

```bash
curl -X POST localhost:8000/knowledge-base/index -H "Content-Type: application/json" \
  -d '{"docs": {"shipping": "Orders ship within 2 business days.", "returns": "Returns accepted within 30 days with receipt."}}'

curl -X POST localhost:8000/ask -H "Content-Type: application/json" \
  -d '{"question": "how long until my order ships?"}'
```

## Test it

```bash
pytest -q
```

10 tests, all passing, no network calls — covers the retriever directly (`tests/test_rag.py`) and the full HTTP surface (`tests/test_main.py`).

## Stack

FastAPI, Pydantic v2, scikit-learn (TF-IDF retrieval), pytest. Designed to drop an OpenAI (or other provider) embeddings client into `OpenAIRetriever` and a real chat-completion call into `compose_answer` for production use.

## About me

Shohel Rana — full-stack developer and AI automation engineer, Bangladesh (remote). I build and run [ShopHelp.chat](https://shophelp.chat) (AI sales chatbot for Shopify/WooCommerce) and [NeverBlast](https://neverblast.com) (AI cold-email outreach) end-to-end, solo — Python/FastAPI, React/Next.js, n8n, LLM integrations. Contact: mrshohelux@gmail.com.
