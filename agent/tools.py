import os
import httpx
from dotenv import load_dotenv, find_dotenv
from langchain_core.tools import tool


_ = load_dotenv(find_dotenv())

API_BASE_URL = os.getenv("API_BASE_URL", f"http://api:{os.getenv('API_PORT', '8000')}")


@tool
def search_knowledge_base(query: str, n_results: int = 5) -> str:
    """Search the knowledge base for documents semantically similar to the query.

    Use this tool whenever you need information from the knowledge base to
    answer the user. Results are ordered by relevance; a lower distance means
    a closer match.

    Args:
        query: Natural language text describing what to look for.
        n_results: Maximum number of documents to return, between 1 and 50.
    """
    try:
        response = httpx.post(
            f"{API_BASE_URL}/search",
            json={"query": query, "n_results": n_results},
            timeout=10.0,
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        return f"Knowledge base search failed: {exc}"

    results = response.json()["results"]
    if not results:
        return f"No documents found for query: {query}"
    return "\n".join(
        f"[{result['id']}] (distance={result['distance']:.4f}) {result['text']}"
        for result in results
    )


toolbox = [search_knowledge_base]
