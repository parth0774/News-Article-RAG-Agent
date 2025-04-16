import os
import json
from typing import Dict, List
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from datetime import datetime

def inspect_vector_store() -> Dict:
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        persist_dir = os.path.join(
            os.path.dirname(__file__),
            "chroma_db"
        )
        
        if not os.path.exists(persist_dir):
            raise FileNotFoundError(f"Vector store directory not found at {persist_dir}")
        
        vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings
        )
        
        collection = vectorstore._collection
        
        docs = collection.get()
        
        stats = {
            "total_documents": len(docs['ids']),
            "total_chunks": len(docs['ids']),
            "embedding_dimension": len(docs['embeddings'][0]) if docs['embeddings'] else 0,
            "metadata_fields": list(set().union(*(set(d.keys()) for d in docs['metadatas']))),
            "categories": list(set(d.get('category', '') for d in docs['metadatas'] if d.get('category'))),
            "date_range": {
                "min": min(d.get('date', '') for d in docs['metadatas'] if d.get('date')),
                "max": max(d.get('date', '') for d in docs['metadatas'] if d.get('date'))
            }
        }
        
        documents = []
        for i in range(len(docs['ids'])):
            doc_id = docs['ids'][i]
            doc_text = docs['documents'][i]
            doc_metadata = docs['metadatas'][i]
            
            documents.append({
                "id": doc_id,
                "content": doc_text,
                "metadata": doc_metadata
            })
        
        return {
            "statistics": stats,
            "documents": documents
        }
        
    except Exception as e:
        return {
            "error": str(e)
        }

def save_results_to_json(results: Dict):
    output_dir = os.path.join(os.path.dirname(__file__), "inspection_results")
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"vector_store_inspection_{timestamp}.json")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    return output_file

def main():
    print("Inspecting vector store...")
    results = inspect_vector_store()
    
    if "error" in results:
        print(f"Error: {results['error']}")
        return
    
    output_file = save_results_to_json(results)
    print(f"\nInspection results saved to: {output_file}")
    print(f"Total documents: {results['statistics']['total_documents']}")
    print(f"Metadata fields: {', '.join(results['statistics']['metadata_fields'])}")

if __name__ == "__main__":
    main() 