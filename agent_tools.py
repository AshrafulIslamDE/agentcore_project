from langchain_core.tools import tool

from vector_database import get_vector_store, search_database


@tool
def search_faq(query:str):
    """
    Returns:
       Relevant faq entries that might answer the question
    """
    query_result,entries=search_database(query)
    return f"found {len(entries)} relevant faq entries:{entries}"

@tool
def reformulate_query(query, aspect):
    """ Reformulate a query to focus on a specific aspect.
    Use this when you need to search for a different angle of the question.

    Args:
    :param query:The query to be reformulated.
    :param aspect: The specific aspect to focus on (e.g., price, activation,troubleshooting, deactivate, bill)
    """
    reformulated_query=f"{aspect} related to {query}"
    results,entries=search_database(reformulated_query)
    return f"Result from {aspect} aspect: {entries}"
