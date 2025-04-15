import os
import warnings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import spacy

warnings.filterwarnings('ignore')

nlp = spacy.load("en_core_web_sm")

class SimpleRetriever:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore

    def semantic_search(self, query, k=2): 
        enhanced_query = f"Find news articles about: {query}"
        results = self.vectorstore.similarity_search_with_score(enhanced_query, k=k)
        return results

def main():
    print("Loading vector store...")
    
    persist_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Vector_Creation_Test", "chroma_db")
    print(f"Looking for vector store in: {persist_dir}")
    
    if not os.path.exists(persist_dir):
        print(f"Error: Vector store directory not found at {persist_dir}")
        return
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    try:
        vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings
        )
        
        collection_count = vectorstore._collection.count()
        print(f"Vector store loaded successfully with {collection_count} documents")
        
    except Exception as e:
        print(f"Error loading vector store: {str(e)}")
        return
    
    retriever = SimpleRetriever(vectorstore)
    
    test_queries = [
        "News on Business"
    ]
    
    print("\nTesting semantic search with small dataset (20 articles)...")
    for query in test_queries:
        print(f"\nQuery: {query}")
        try:
            results = retriever.semantic_search(query)
            
            if not results:
                print("No results found for this query")
                continue
                
            print("\nRetrieved Documents:")
            for doc, score in results:
                print(f"\nSimilarity Score: {score:.4f}") 
                print(f"Category: {doc.metadata.get('category', 'N/A')}")
                print(f"Headline: {doc.metadata.get('headline', 'N/A')}")
                print(f"Date: {doc.metadata.get('date', 'N/A')}")
                print(f"Full Content Available: {'Yes' if doc.metadata.get('has_full_content', False) else 'No'}")
                if doc.metadata.get('link'):
                    print(f"Link: {doc.metadata['link']}")
                print("Entities found in document:")
                print(doc.metadata.get('entities', 'No entities found'))
                print("-" * 50)
        except Exception as e:
            print(f"Error processing query '{query}': {str(e)}")

if __name__ == "__main__":
    main() 