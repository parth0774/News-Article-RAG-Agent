import os
import warnings
import time
from typing import Dict, List, Any, Optional
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
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
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import RetrievalQA
import logging
from conversation_manager import conversation_manager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('RAG_Agent')

warnings.filterwarnings('ignore')

load_dotenv()

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
                model_name=os.getenv("LLM_MODEL", "gpt-4o-mini"),
                temperature=float(os.getenv("LLM_TEMPERATURE", 0.1))
            )
            
            self.chain = self._setup_rag_chain()
            
            self.conversation_history = []
            
            self.qa_prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="""You are a helpful news assistant that provides accurate and relevant information from the knowledge base.
                Use the provided context to answer questions. If you don't know the answer, say so.
                Consider the conversation history to provide contextually relevant responses.
                Extract and highlight important entities (people, organizations, locations, dates) in your response."""),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
                ("assistant", "Let me search the knowledge base and provide you with relevant information."),
            ])
            
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 3}
                ),
                return_source_documents=True
            )
            
            logger.info("RAG system initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing RAG system: {str(e)}")
            raise
    
    def _load_vector_store(self) -> Chroma:
        persist_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            os.getenv("VECTOR_STORE_DIR", r"C:\Users\parth\Desktop\test2\chroma_db")
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
            weights=[0.5, 0.5] 
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
            6. Consider the article's category and date when providing context
            7. Include the author's name when citing sources
            
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
    
    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        doc = self.nlp(text)
        entities = {
            "PERSON": [],
            "ORG": [],
            "GPE": [],  # Geo-Political Entity
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
    
    def query(self, question: str) -> Dict:
        try:
            question_entities = self._extract_entities(question)
            
            docs = self.ensemble_retriever.get_relevant_documents(question)
            
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

    def _update_conversation_history(self, message: BaseMessage):
        self.conversation_history.append(message)
        if len(self.conversation_history) > 5:
            self.conversation_history = self.conversation_history[-5:]
    
    def process_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        try:
            logger.info(f"Processing query: {query}")
            
            entities = self._extract_entities(query)
            logger.info(f"Extracted entities: {entities}")
            
            conversation_id = context.get("conversation_id", "default")
            chat_history = conversation_manager.get_conversation(conversation_id)
            
            chain_input = {
                "question": query,
                "chat_history": chat_history
            }
            
            if context:
                chain_input.update(context)
            
            result = self.qa_chain.invoke(chain_input)
            
            answer = result.get("result", "I couldn't find an answer to your question.")
            source_docs = result.get("source_documents", [])
            
            response = f"{answer}\n\nSources:\n"
            for i, doc in enumerate(source_docs, 1):
                response += f"{i}. {doc.metadata.get('source', 'Unknown source')}\n"
            
            response += "\nKey Entities:\n"
            for entity_type, entity_list in entities.items():
                if entity_list:
                    response += f"{entity_type}: {', '.join(entity_list)}\n"
            
            logger.info(f"Generated response with {len(source_docs)} sources and {len(entities)} entity types")
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            error_message = f"I encountered an error while processing your query: {str(e)}"
            return error_message
    
    def get_conversation_history(self) -> List[BaseMessage]:
        return self.conversation_history
    
    def clear_conversation_history(self):
        self.conversation_history = []

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