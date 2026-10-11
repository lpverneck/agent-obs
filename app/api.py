import chromadb
from fastapi import FastAPI


app = FastAPI()
chroma_client = chromadb.HttpClient(host="vector-db", port=8000)
chroma_client.heartbeat()

collection = chroma_client.get_or_create_collection(name="my_collection")

collection.upsert(
    ids=["id1", "id2", "id3", "id4", "id5"],
    documents=[
        "This is a document about pineapple",
        "This is a document about oranges",
        "This a document abou surfers on Hawaii",
        "This is a random document about nothing in particular",
        "This is a curiosity about the volcans in Hawaii areas",
    ],
)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: str | None = None):
    return {"item_id": item_id, "q": q}


@app.get("/search")
def search_documents(q: str | None = None):
    if not q:
        return {"error": "Query parameter 'q' is required!"}
    results = collection.query(query_texts=[q], n_results=2)
    return results
