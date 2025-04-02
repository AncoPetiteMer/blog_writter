import logging
import os
import re
from typing import List

logger = logging.getLogger("seo_blog_tools")


# TODO: make advanced query to get tpics from last week
def search_with_tavily(topic: str) -> List[str]:
    """Recherche de mots-clés avec Tavily à partir du topic."""
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        result = client.search(query=f"{topic} SEO keywords", search_depth="advanced", max_results=3)
        content = " ".join([res.get("content", "") for res in result.get("results", [])])
        keywords = re.findall(r'\b[\w\s]{4,}\b', content)
        return list(set(keywords))[:5]
    except Exception as e:
        logger.warning(f"Tavily error: {e}")
        return []


# TODO: make advanced query to get context of last week topic
def search_with_wikipedia(topic: str, lang: str = "en") -> List[str]:
    """Mots-clés SEO extraits d’un résumé Wikipedia."""
    try:
        import wikipediaapi
        wiki = wikipediaapi.Wikipedia(lang)
        page = wiki.page(topic)
        if page.exists():
            words = re.findall(r'\b[\w\s]{4,}\b', page.summary)
            return list(set(words))[:5]
    except Exception as e:
        logger.warning(f"Wikipedia error: {e}")
    return []


def search_with_exa(topic: str) -> List[str]:
    """(Placeholder) Fausse recherche Exa – à implémenter avec API réelle."""
    return [f"{topic} analysis", f"{topic} insights", f"{topic} examples"]
