from schemas import KBDocument, SearchRequest, SearchResponse, SearchResult

import logging
import uuid
import chromadb
from fastapi import FastAPI
from opentelemetry import metrics


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

meter = metrics.get_meter(__name__)
search_top_distance = meter.create_histogram(
    name="kb.search.top_distance",
    description="Distance of the most relevant document returned by a search",
    explicit_bucket_boundaries_advisory=[
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.75,
        1.0,
        1.25,
        1.5,
        2.0,
    ],
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
    results = collection.query(query_texts=[search.query], n_results=search.n_results)
    if results["distances"][0]:
        search_top_distance.record(results["distances"][0][0])
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
