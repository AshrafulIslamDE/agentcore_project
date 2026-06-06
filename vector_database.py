from langchain_community.vectorstores import FAISS

from constraints import VECTOR_DATABASE_FOLDER_PATH
from embedding import get_embeddings

def get_vector_store():
    embeddings = get_embeddings()
    store = FAISS.load_local(
        VECTOR_DATABASE_FOLDER_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return store

def search_database(query, k=3):

    results = get_vector_store().similarity_search(query, k=k)
    entries=""
    if not results:
        print("no data found stored in the database")
        return [],entries
    for i,entry in enumerate(results,1):
        entries+=f"entry {i}: {entry.page_content}\n"

    return results,entries
