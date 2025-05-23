import os
import warnings
import time
from typing import Dict, List, Any, Optional
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from newspaper import Article
from urllib.parse import urlparse
import spacy
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('LinkedIn_Agent')

warnings.filterwarnings('ignore')

load_dotenv()

class LinkedInAgent:
    def __init__(self):
        try:
            self.llm = ChatOpenAI(
                model=os.getenv("LLM_MODEL"),
                temperature=float(os.getenv("LLM_TEMPERATURE", "0.7"))
            )
            
            self.prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a professional LinkedIn post generator.
                Create engaging, professional, and informative LinkedIn posts based on the provided content.
                
                Guidelines:
                1. Start with an attention-grabbing hook
                2. Use professional language
                3. Include relevant hashtags
                4. Keep the tone positive and engaging
                5. Add a call to action when appropriate
                6. Consider the target audience and industry context
                7. Maintain consistency with previous posts in the conversation
                8. Use the provided context to add relevant details and insights
                9. Include statistics or data points when available
                10. End with a thought-provoking question or call to action"""),
                ("human", "{content}")
            ])
            
            logger.info("LinkedIn agent initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing LinkedIn agent: {str(e)}")
            raise
    
    def generate_post(self, content: str, context: Optional[Dict[str, Any]] = None) -> str:
        try:
            logger.info("Generating LinkedIn post...")
            
            chain_input = {
                "content": content
            }
            
            if context:
                chain_input.update(context)
            
            chain = (
                {"content": RunnablePassthrough()}
                | self.prompt
                | self.llm
                | StrOutputParser()
            )
            
            post = chain.invoke(content)
            
            logger.info("LinkedIn post generated successfully")
            return post
            
        except Exception as e:
            logger.error(f"Error generating LinkedIn post: {str(e)}")
            return "I encountered an error while generating your LinkedIn post."

class RAGSystem:
    def __init__(self):
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2"),
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            self.nlp = spacy.load("en_core_web_sm")
            
            self.vectorstore = self._load_vector_store()
            
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            
            self.ensemble_retriever = self._setup_retrievers()
            
            self.llm = ChatOpenAI(
                model_name=os.getenv("LLM_MODEL"),
                temperature=float(os.getenv("LLM_TEMPERATURE", 0.1))
            )
            
            self.chain = self._setup_rag_chain()
            
        except Exception as e:
            raise Exception(f"Failed to initialize RAG system: {str(e)}")
    
    def _load_vector_store(self) -> Chroma:
        persist_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "Create_Vectorstore",
            "chroma_db"
        )
        
        if not os.path.exists(persist_dir):
            raise FileNotFoundError(f"Vector store directory not found at {persist_dir}")
        
        try:
            vectorstore = Chroma(
                persist_directory=persist_dir,
                embedding_function=self.embeddings
            )
            
            collection = vectorstore._collection
            docs = collection.get()
            if not docs['ids']:
                raise ValueError("No documents found in the vector store")
                
            return vectorstore
            
        except Exception as e:
            raise Exception(f"Failed to load vector store: {str(e)}")
    
    def _setup_retrievers(self) -> EnsembleRetriever:
        try:
            collection = self.vectorstore._collection
            docs = collection.get()
            
            if not docs['ids']:
                raise ValueError("No documents available for retrieval")
            
            bm25_docs = []
            bm25_metadatas = []
            
            for i in range(len(docs['ids'])):
                doc_text = docs['documents'][i]
                doc_metadata = docs['metadatas'][i]
                
                chunks = self.text_splitter.split_text(doc_text)
                
                bm25_docs.extend(chunks)
                bm25_metadatas.extend([doc_metadata] * len(chunks))
            
            if not bm25_docs:
                raise ValueError("No text chunks available for BM25 retrieval")
            
            bm25_retriever = BM25Retriever.from_texts(
                bm25_docs,
                metadatas=bm25_metadatas
            )
            bm25_retriever.k = min(2, len(bm25_docs))  
            
            vector_retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": min(2, len(docs['ids']))}  
            )
            
            ensemble_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, vector_retriever],
                weights=[0.5, 0.5] 
            )
            
            return ensemble_retriever
            
        except Exception as e:
            raise Exception(f"Failed to setup retrievers: {str(e)}")
    
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
        try:
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", """You are a professional LinkedIn content creator. Create an engaging LinkedIn post based on the provided news articles.
                
                Instructions:
                1. Start with an attention-grabbing hook using relevant emojis
                2. Present the key insights in a clear, professional manner
                3. Add your professional perspective or analysis
                4. Include relevant statistics or data points
                5. Break up text with line breaks for readability
                6. End with a thought-provoking question or call to action
                7. Keep the tone professional but engaging
                8. Use emojis sparingly and appropriately
                9. Include relevant hashtags
                10. Always cite your sources
                
                Format:
                [Hook with emoji]
                
                [Main content with line breaks]
                
                [Professional analysis]
                
                [Call to action or question]
                
                [Relevant hashtags]
                
                [Source attribution]
                
                Context:
                {context}
                
                Topic: {question}
                
                Create a LinkedIn post that is informative, engaging, and professional."""),
                ("human", "{question}")
            ])
            
            return (
                {"context": self.ensemble_retriever, 
                 "question": RunnablePassthrough()}
                | prompt_template
                | self.llm
                | StrOutputParser()
            )
            
        except Exception as e:
            raise Exception(f"Failed to setup RAG chain: {str(e)}")
    
    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        try:
            doc = self.nlp(text)
            entities = {
                "PERSON": [],
                "ORG": [],
                "GPE": [],  
                "DATE": [],
                "TIME": [],
                "MONEY": [],
                "PERCENT": [],
                "EVENT": []
            }
            
            for ent in doc.ents:
                if ent.label_ in entities:
                    entities[ent.label_].append(ent.text)
            
            return entities
            
        except Exception as e:
            print(f"Error extracting entities: {str(e)}")
            return {key: [] for key in ["PERSON", "ORG", "GPE", "DATE", "TIME", "MONEY", "PERCENT", "EVENT"]}
    
    def query(self, question: str) -> Dict:
        try:
            if not question or not question.strip():
                raise ValueError("Question cannot be empty")
            
            question_entities = self._extract_entities(question)
            
            docs = self.ensemble_retriever.get_relevant_documents(question)
            
            if not docs:
                return {
                    "error": "No relevant documents found",
                    "answer": None,
                    "sources": None,
                    "question_entities": question_entities
                }
            
            enhanced_docs = []
            for doc in docs:
                if "link" in doc.metadata and doc.metadata["link"]:
                    print(f"Fetching additional content from: {doc.metadata['link']}")
                    additional_content = self._fetch_article_content(doc.metadata["link"])
                    if additional_content:
                        content_entities = self._extract_entities(additional_content)
                        enhanced_content = f"{doc.page_content}\n\nAdditional content from article:\n{additional_content}"
                        enhanced_docs.append({
                            "content": enhanced_content,
                            "metadata": doc.metadata,
                            "entities": content_entities
                        })
                    else:
                        enhanced_docs.append({
                            "content": doc.page_content,
                            "metadata": doc.metadata,
                            "entities": self._extract_entities(doc.page_content)
                        })
                else:
                    enhanced_docs.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "entities": self._extract_entities(doc.page_content)
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
                    "authors": doc["metadata"].get("authors", "N/A"),
                    "short_description": doc["metadata"].get("short_description", "N/A"),
                    "content_preview": doc["content"][:200] + "...",
                    "entities": doc["entities"]
                }
                sources.append(source_info)
            
            return {
                "answer": answer,
                "sources": sources,
                "question_entities": question_entities
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "answer": None,
                "sources": None,
                "question_entities": None
            }

def main():
    print("Initializing RAG system with hybrid retrieval...")
    
    try:
        rag = RAGSystem()
        
        question = "Find news about american airlines"
        
        print(f"\nQuestion: {question}")
        result = rag.query(question)
        
        if result.get("error"):
            print(f"Error: {result['error']}")
            return
            
        print("\nGenerated LinkedIn Post:")
        print("-" * 50)
        print(result["answer"])
        print("-" * 50)
        
        print("\nSources:")
        for i, source in enumerate(result["sources"], 1):
            print(f"\nSource {i}:")
            print(f"Headline: {source['headline']}")
            print(f"Category: {source['category']}")
            print(f"Date: {source['date']}")
            print(f"Authors: {source['authors']}")
            if source['link']:
                print(f"Link: {source['link']}")
            print(f"Description: {source['short_description']}")
            print(f"Content Preview: {source['content_preview']}")
            print("-" * 50)
            
    except Exception as e:
        print(f"Error initializing RAG system: {str(e)}")

if __name__ == "__main__":
    main() 