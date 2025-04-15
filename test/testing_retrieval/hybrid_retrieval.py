import os
import spacy
from typing import List, Dict
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

class HybridRetriever:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.vectorstore = self._load_vector_store()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.ensemble_retriever = self._setup_retrievers()
    
    def _load_vector_store(self) -> Chroma:
        persist_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "vector_creation/chroma_db"
        )
        if not os.path.exists(persist_dir):
            raise FileNotFoundError(f"Vector store directory not found at {persist_dir}")
        return Chroma(
            persist_directory=persist_dir,
            embedding_function=self.embeddings
        )
    
    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        doc = self.nlp(text)
        entities = {
            "PERSON": [],
            "GPE": [],
            "LOC": [],
            "ORG": [],
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
    
    def retrieve_documents(self, query: str) -> Dict:
        try:
            query_entities = self._extract_entities(query)
            docs = self.ensemble_retriever.get_relevant_documents(query)
            results = []
            for doc in docs:
                doc_entities = self._extract_entities(doc.page_content)
                entity_overlap = {}
                for entity_type in query_entities:
                    if query_entities[entity_type]:
                        overlap = len(set(query_entities[entity_type]) & set(doc_entities[entity_type]))
                        entity_overlap[entity_type] = overlap
                result = {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "entities": doc_entities,
                    "entity_overlap": entity_overlap
                }
                results.append(result)
            return {
                "query_entities": query_entities,
                "results": results
            }
        except Exception as e:
            return {
                "error": str(e),
                "results": None
            }

def main():
    print("Initializing hybrid retriever...")
    try:
        retriever = HybridRetriever()
        queries = [
            "Find news about united airlines",
        ]
        for query in queries:
            print(f"\nQuery: {query}")
            result = retriever.retrieve_documents(query)
            if result.get("error"):
                print(f"Error: {result['error']}")
                continue
            print("\nQuery Entities:")
            for entity_type, entities in result["query_entities"].items():
                if entities:
                    print(f"{entity_type}: {', '.join(entities)}")
            print("\nResults:")
            for i, doc in enumerate(result["results"], 1):
                print(f"\nDocument {i}:")
                print(f"Headline: {doc['metadata'].get('headline', 'N/A')}")
                print(f"Category: {doc['metadata'].get('category', 'N/A')}")
                print(f"Date: {doc['metadata'].get('date', 'N/A')}")
                print("\nEntity Overlap:")
                for entity_type, overlap in doc["entity_overlap"].items():
                    if overlap > 0:
                        print(f"{entity_type}: {overlap} matches")
                print("\nContent Preview:")
                print(doc["content"][:200] + "...")
                print("-" * 50)
    except Exception as e:
        print(f"Error initializing hybrid retriever: {str(e)}")

if __name__ == "__main__":
    main()
