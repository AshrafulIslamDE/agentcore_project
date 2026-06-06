import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from constraints import VECTOR_DATABASE_FOLDER_PATH
from embedding import get_embeddings


def load_faq_csv():
    df = pd.read_csv('lauki_qna.csv')
    df = df.apply(lambda col: col.str.strip() if col.dtype == 'object' else col)
    records = df[['question', 'answer']].to_dict(orient='records')
    return records



def make_documents(records):
    # one Document per Q&A pair
    docs = [
        Document(
            page_content=f"Question: {r['question']}\nAnswer: {r['answer']}",
            metadata={"question": r["question"], "answer": r["answer"]},
        )
        for r in records
    ]
    return docs


def store_chunks(docs):
    store = FAISS.from_documents(docs, get_embeddings())
    store.save_local(VECTOR_DATABASE_FOLDER_PATH)
    return store


def handle_documents():
    records = load_faq_csv()
    docs = make_documents(records)
    store_chunks(docs)

def ask_question(query, k=1):
    embeddings = get_embeddings()
    store = FAISS.load_local(
        "faq_index",
        embeddings,
        allow_dangerous_deserialization=True,
    )

    results = store.similarity_search(query, k=k)

    for i, doc in enumerate(results, 1):
        print(f"\n--- Result {i} ---")
        print(doc.metadata.get("answer", doc.page_content))

    return results

if __name__ == '__main__':
    # handle_documents()
    query="how activate sim?"
    results=ask_question(query)
    print(results)