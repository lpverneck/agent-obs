from schemas import KBDocument, SearchRequest, SearchResponse, SearchResult

import json
import logging
import uuid
import chromadb
from fastapi import FastAPI
from opentelemetry import baggage, metrics, trace
from opentelemetry.processor.baggage import BaggageSpanProcessor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import SpanKind


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# request identifiers propagated as baggage from the agent service
BAGGAGE_KEYS = ("session.id", "user.id")

# copy the baggage identifiers onto every span this service starts
tracer_provider = trace.get_tracer_provider()
if isinstance(
    tracer_provider, TracerProvider
):  # only set under opentelemetry-instrument
    tracer_provider.add_span_processor(
        BaggageSpanProcessor(lambda key: key in BAGGAGE_KEYS)
    )

tracer = trace.get_tracer(__name__)

meter = metrics.get_meter(__name__)
search_documents_counter = meter.create_counter(
    name="kb.search.documents",
    unit="{document}",
    description="Number of documents returned by searches",
)

CHROMA_HOST = "vector-db"
CHROMA_PORT = 8000

app = FastAPI()
chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
chroma_client.heartbeat()

collection = chroma_client.get_or_create_collection(name="my_collection")

collection.upsert(
    ids=["id1", "id2", "id3", "id4", "id5"],
    documents=[
        "This is a document about pineapple",
        "This is a document about oranges",
        "This a document about surfers on Hawaii",
        "This is a random document about nothing in particular",
        "This is a curiosity about the volcans in Hawaii areas",
    ],
)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/search")
def search_documents(search: SearchRequest) -> SearchResponse:
    logger.info(
        "search executed: query=%s n_results=%d", search.query, search.n_results
    )
    # last hop of the flow: confirm the request identifiers arrived via the baggage header
    logger.info(
        "baggage received: session.id=%s user.id=%s",
        baggage.get_baggage("session.id"),
        baggage.get_baggage("user.id"),
    )
    # retrieval span following the GenAI semantic conventions
    # (query text and documents are Opt-In attributes, they may hold sensitive data)
    with tracer.start_as_current_span(
        f"retrieval {collection.name}",
        kind=SpanKind.CLIENT,
        attributes={
            "gen_ai.operation.name": "retrieval",
            "gen_ai.data_source.id": collection.name,
            "gen_ai.retrieval.top_k": search.n_results,
            "gen_ai.retrieval.query.text": search.query,
            "server.address": CHROMA_HOST,
            "server.port": CHROMA_PORT,
        },
    ) as span:
        try:
            results = collection.query(
                query_texts=[search.query], n_results=search.n_results
            )
        except Exception as exc:
            span.set_attribute("error.type", type(exc).__qualname__)
            raise
        search_results = [
            SearchResult(id=doc_id, text=text, distance=distance)
            for doc_id, text, distance in zip(
                results["ids"][0], results["documents"][0], results["distances"][0]
            )
        ]
        # chroma returns a distance (lower is closer), not a relevance score,
        # so it goes in an extra property instead of the standard `score`
        span.set_attribute(
            "gen_ai.retrieval.documents",
            json.dumps(
                [
                    {
                        "id": result.id,
                        "distance": result.distance,
                        "content": result.text,
                    }
                    for result in search_results
                ]
            ),
        )
    search_documents_counter.add(len(search_results))
    return SearchResponse(results=search_results)


@app.post("/documents", status_code=201)
def add_document(document: KBDocument):
    doc_id = document.id or str(uuid.uuid4())
    collection.add(ids=[doc_id], documents=[document.text])
    return {"id": doc_id, "text": document.text}
