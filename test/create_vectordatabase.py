import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever 
from langchain.retrievers import EnsembleRetriever

df = pd.read_json(
    r"C:\Users\parth\Desktop\Projects\News-Article-RAG-Agent\News-Article-RAG-Agent\test\Dataset\News_Category_Dataset_v3.json",
    lines=True
)
df["content"] = (
    "Headline: " + df["headline"].astype(str) + ". " +
    "Short Description: " + df["short_description"].astype(str) + ". " +
    "Author(s): " + df["authors"].astype(str) + ". " +
    "Date: " + pd.to_datetime(df["date"]).dt.strftime("%B %d, %Y") + ". " +
    "Link: " + df["link"].astype(str) + ". " +
    "Category: " + df["category"].astype(str) + "."
)
subset_df = df[:10]  

texts = subset_df["content"].tolist()
metadatas = subset_df[["headline", "category", "authors", "date", "link"]].to_dict(orient="records")

bm25_retriever = BM25Retriever.from_texts(texts, metadatas=metadatas)
bm25_retriever.k = 1

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
faiss_store = FAISS.from_texts(texts, embedding_model, metadatas=metadatas)
faiss_retriever = faiss_store.as_retriever(search_kwargs={"k": 1})

hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever],
    weights=[0.2, 0.8]
)

query = "What are Covid Boosters?"

results = hybrid_retriever.get_relevant_documents(query)

print(f"\n🔍 Query: {query}\n")
for i, doc in enumerate(results, 1):
    print(f"--- Result {i} ---")
    print("Headline:", doc.metadata.get('headline'))
    print("Category:", doc.metadata.get('category'))
    print("Date:", doc.metadata.get('date'))
    print("Snippet:", doc.page_content[:300], "...\n")
