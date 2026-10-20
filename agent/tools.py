import os
import httpx
from dotenv import load_dotenv, find_dotenv
from langchain_core.tools import tool


_ = load_dotenv(find_dotenv())

API_BASE_URL = os.getenv("API_BASE_URL", f"http://api:{os.getenv('API_PORT', '8000')}")


@tool
def search_knowledge_base(query: str, n_results: int = 5) -> str:
    """Search the IT support knowledge base for articles semantically similar to the query.

    The knowledge base holds the company's official IT how-to guides and
    troubleshooting articles. Use this tool for any employee question about
    company devices, accounts, software or network access. Results are ordered
    by relevance; a lower distance means a closer match.

    Args:
        query: Natural language description of the IT issue or task, e.g.
            "reset a forgotten PIN" or "set up VPN to work from home".
        n_results: Maximum number of articles to return, between 1 and 50.
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
