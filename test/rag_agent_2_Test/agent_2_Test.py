import os
import warnings
import time
from typing import Dict, List, Optional
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from newspaper import Article
from urllib.parse import urlparse
from dotenv import load_dotenv


warnings.filterwarnings('ignore')

load_dotenv()

class RAGSystem:
    def __init__(self, openai_api_key: Optional[str] = None):
        self._setup_openai(openai_api_key)
        self.embeddings = self._setup_embeddings()
        self.vectorstore = self._load_vector_store()
        self.llm = self._setup_llm()
        self.chain = self._setup_rag_chain()
    
    def _setup_openai(self, api_key: Optional[str]) -> None:
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        elif not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OpenAI API key not found. Please provide it in .env file or as parameter.")
    
    def _setup_embeddings(self) -> HuggingFaceEmbeddings:
        return HuggingFaceEmbeddings(
            model_name=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2"),
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    
    def _setup_llm(self) -> ChatOpenAI:
        return ChatOpenAI(
            model_name=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            temperature=float(os.getenv("LLM_TEMPERATURE", 0.1))
        )
    
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
    
    def _fetch_article_content(self, url: str) -> str:
        try:
            if not url or not urlparse(url).scheme:
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
            ("system", """You are a helpful news assistant. Your task is to analyze the provided news articles and answer questions about them.
            
            Instructions:
            1. Carefully read and analyze all the provided context
            2. If the context contains relevant information, use it to answer the question
            3. If the context is not directly about the topic, look for related information
            4. Always cite your sources using the provided metadata
            5. If you find multiple relevant articles, combine the information
            6. If you truly cannot find any relevant information, say so
            
            Context:
            {context}
            
            Question: {question}
            
            Based on the above context, provide a detailed answer. Include specific details from the articles and cite your sources."""),
            ("human", "{question}")
        ])
        
        return (
            {"context": self.vectorstore.as_retriever(search_kwargs={"k": 2}), 
             "question": RunnablePassthrough()}
            | prompt_template
            | self.llm
            | StrOutputParser()
        )
    
    def query(self, question: str) -> Dict:
        try:
            docs = self.vectorstore.similarity_search(question, k=2)
            enhanced_docs = []
            for doc in docs:
                if doc.metadata.get("link"):
                    print(f"\nFetching additional content from: {doc.metadata['link']}")
                    additional_content = self._fetch_article_content(doc.metadata["link"])
                    if additional_content:
                        enhanced_content = f"{doc.page_content}\n\nADDITIONAL CONTENT:\n{additional_content}"
                        enhanced_doc = doc.copy()
                        enhanced_doc.page_content = enhanced_content
                        enhanced_docs.append(enhanced_doc)
                        time.sleep(1)
                    else:
                        enhanced_docs.append(doc)
                else:
                    enhanced_docs.append(doc)
            
            answer = self.chain.invoke(question)
            
            sources = self._prepare_sources(enhanced_docs)
            
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
    
    def _prepare_sources(self, docs: List) -> List[Dict]:
        sources = []
        for doc in docs:
            source_info = {
                "headline": doc.metadata.get("headline", "N/A"),
                "category": doc.metadata.get("category", "N/A"),
                "date": doc.metadata.get("date", "N/A"),
                "link": doc.metadata.get("link", "N/A"),
                "content": doc.page_content[:200] + "..." 
            }
            sources.append(source_info)
        return sources

def main():
    print("Initializing RAG system...")
    
    try:
        rag = RAGSystem()
        
        test_questions = [
            "Find news about dead cases"
        ]
        
        print("\nTesting RAG system...")
        for question in test_questions:
            print(f"\nQuestion: {question}")
            result = rag.query(question)
            
            if result.get("error"):
                print(f"Error: {result['error']}")
                continue
                
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
                print(f"Content Preview: {source['content']}")
            print("-" * 50)
            
    except Exception as e:
        print(f"Error initializing RAG system: {str(e)}")

if __name__ == "__main__":
    main()