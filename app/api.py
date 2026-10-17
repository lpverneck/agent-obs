from schemas import KBDocument, SearchRequest, SearchResponse, SearchResult

import logging
import uuid
import chromadb
from fastapi import FastAPI
from opentelemetry import baggage, metrics, trace
from opentelemetry.processor.baggage import BaggageSpanProcessor
from opentelemetry.sdk.trace import TracerProvider


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# request identifiers propagated as baggage from the agent service
BAGGAGE_KEYS = ("session.id", "user.id")

# copy the baggage identifiers onto every span this service starts
tracer_provider = trace.get_tracer_provider()
if isinstance(tracer_provider, TracerProvider):  # only set under opentelemetry-instrument
    tracer_provider.add_span_processor(
        BaggageSpanProcessor(lambda key: key in BAGGAGE_KEYS)
    )

meter = metrics.get_meter(__name__)
search_documents_counter = meter.create_counter(
    name="kb.search.documents",
    unit="{document}",
    description="Number of documents returned by searches",
)

app = FastAPI()
chroma_client = chromadb.HttpClient(host="vector-db", port=8000)
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
    results = collection.query(query_texts=[search.query], n_results=search.n_results)
    search_documents_counter.add(len(results["ids"][0]))
    return SearchResponse(
        results=[
            SearchResult(id=doc_id, text=text, distance=distance)
            for doc_id, text, distance in zip(
                results["ids"][0], results["documents"][0], results["distances"][0]
            )
        ]
    )


@app.post("/documents", status_code=201)
def add_document(document: KBDocument):
    doc_id = document.id or str(uuid.uuid4())
    collection.add(ids=[doc_id], documents=[document.text])
    return {"id": doc_id, "text": document.text}
