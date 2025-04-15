import os
import warnings
import time
from typing import Dict, List
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from newspaper import Article
from urllib.parse import urlparse

warnings.filterwarnings('ignore')
load_dotenv()

class RAGSystem:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2"),
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.vectorstore = self._load_vector_store()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.ensemble_retriever = self._setup_retrievers()
        self.llm = ChatOpenAI(
            model_name=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            temperature=float(os.getenv("LLM_TEMPERATURE", 0.1))
        )
        self.chain = self._setup_rag_chain()
    
    def _load_vector_store(self) -> Chroma:
        persist_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            os.getenv("VECTOR_STORE_DIR", "vector_creation/chroma_db")
        )
        if not os.path.exists(persist_dir):
            raise FileNotFoundError(f"Vector store directory not found at {persist_dir}")
        return Chroma(
            persist_directory=persist_dir,
            embedding_function=self.embeddings
        )
    
    def _setup_retrievers(self) -> EnsembleRetriever:
        collection = self.vectorstore._collection
        docs = collection.get()
        bm25_docs = []
        bm25_metadatas = []
        for i in range(len(docs['ids'])):
            doc_text = docs['documents'][i]
            doc_metadata = docs['metadatas'][i]
            chunks = self.text_splitter.split_text(doc_text)
            bm25_docs.extend(chunks)
            bm25_metadatas.extend([doc_metadata] * len(chunks))
        bm25_retriever = BM25Retriever.from_texts(
            bm25_docs,
            metadatas=bm25_metadatas
        )
        bm25_retriever.k = 2
        vector_retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": 2}
        )
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.8, 0.2]
        )
        return ensemble_retriever
    
    def _fetch_article_content(self, url: str) -> str:
        try:
            parsed_url = urlparse(url)
            if not all([parsed_url.scheme, parsed_url.netloc]):
                return ""
            article = Article(url)
            article.download()
            article.parse()
            return article.text
        except Exception as e:
            print(f"Error fetching article from {url}: {str(e)}")
            return ""
    
    def _setup_rag_chain(self):
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful news assistant. Analyze the provided news articles to answer the question.
            
            Instructions:
            1. Use the provided context to answer the question
            2. If the context contains relevant information, use it
            3. Always cite your sources using the provided metadata
            4. If you find multiple relevant articles, combine the information
            5. If a source has additional content from its link, use that information too
            
            Context:
            {context}
            
            Question: {question}
            
            Provide a detailed answer based on all available information. Include specific details and cite your sources."""),
            ("human", "{question}")
        ])
        return (
            {"context": self.ensemble_retriever, 
             "question": RunnablePassthrough()}
            | prompt_template
            | self.llm
            | StrOutputParser()
        )
    
    def query(self, question: str) -> Dict:
        try:
            docs = self.ensemble_retriever.get_relevant_documents(question)
            enhanced_docs = []
            for doc in docs:
                if "link" in doc.metadata and doc.metadata["link"]:
                    print(f"Fetching additional content from: {doc.metadata['link']}")
                    additional_content = self._fetch_article_content(doc.metadata["link"])
                    if additional_content:
                        enhanced_content = f"{doc.page_content}\n\nAdditional content from article:\n{additional_content}"
                        enhanced_docs.append({
                            "content": enhanced_content,
                            "metadata": doc.metadata
                        })
                    else:
                        enhanced_docs.append({
                            "content": doc.page_content,
                            "metadata": doc.metadata
                        })
                else:
                    enhanced_docs.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    })
                time.sleep(1)
            answer = self.chain.invoke(question)
            sources = []
            for doc in enhanced_docs:
                source_info = {
                    "headline": doc["metadata"].get("headline", "N/A"),
                    "category": doc["metadata"].get("category", "N/A"),
                    "date": doc["metadata"].get("date", "N/A"),
                    "link": doc["metadata"].get("link", "N/A"),
                    "content_preview": doc["content"][:200] + "..."
                }
                sources.append(source_info)
            return {
                "answer": answer,
                "sources": sources
            }
        except Exception as e:
            return {
                "error": str(e),
                "answer": None,
                "sources": None
            }

def main():
    print("Initializing RAG system with hybrid retrieval...")
    try:
        rag = RAGSystem()
        question = "Find news about americal airlines"
        print(f"\nQuestion: {question}")
        result = rag.query(question)
        if result.get("error"):
            print(f"Error: {result['error']}")
            return
        print("\nAnswer:")
        print(result["answer"])
        print("\nSources:")
        for i, source in enumerate(result["sources"], 1):
            print(f"\nSource {i}:")
            print(f"Headline: {source['headline']}")
            print(f"Category: {source['category']}")
            print(f"Date: {source['date']}")
            if source['link']:
                print(f"Link: {source['link']}")
            print(f"Content Preview: {source['content_preview']}")
        print("-" * 50)
    except Exception as e:
        print(f"Error initializing RAG system: {str(e)}")

if __name__ == "__main__":
    main()
