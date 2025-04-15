import json
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from newspaper import Article
import time
from urllib.parse import urlparse
import spacy
from tqdm import tqdm

nlp = spacy.load("en_core_web_sm")

def extract_entities(text):
    doc = nlp(text)
    entities = {
        'PERSON': [],
        'ORG': [],
        'GPE': [],
        'DATE': [],
        'TIME': []
    }
    for ent in doc.ents:
        if ent.label_ in entities:
            entities[ent.label_].append(ent.text)
    return entities

def entities_to_string(entities):
    parts = []
    for entity_type, entity_list in entities.items():
        if entity_list:
            parts.append(f"{entity_type}: {', '.join(entity_list)}")
    return " | ".join(parts)

def fetch_article_content(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        print(f"Error fetching article from {url}: {str(e)}")
        return None

def process_news_data(json_path, limit=20):
    print(f"Processing first {limit} records from the dataset...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f.readlines()[:limit]]
    
    documents = []
    
    for item in tqdm(data, desc="Processing articles"):
        article_content = ""
        if item['link'] and urlparse(item['link']).scheme:
            print(f"\nFetching content from: {item['link']}")
            article_content = fetch_article_content(item['link'])
            time.sleep(1)  
        
        text_parts = [
            f"CATEGORY: {item['category']}",
            f"HEADLINE: {item['headline']}",
            f"AUTHORS: {', '.join(item['authors'])}",
            f"DESCRIPTION: {item['short_description']}"
        ]
        
        if article_content:
            text_parts.append(f"CONTENT: {article_content}")
        
        text = "\n\n".join(text_parts)
        
        entities = extract_entities(text)
        entities_str = entities_to_string(entities)
        
        metadata = {
            'category': item['category'],
            'headline': item['headline'],
            'authors': ', '.join(item['authors']),
            'link': item['link'],
            'date': item['date'],
            'has_full_content': bool(article_content),
            'entities': entities_str
        }
        
        documents.append(Document(page_content=text, metadata=metadata))
    
    return documents

def main():
    print("Initializing vector store creation...")
    
    print("Loading embeddings model...")
    model_name = "sentence-transformers/all-mpnet-base-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    json_path = r"C:\Users\parth\Desktop\Projects\News-Article-RAG-Agent\News-Article-RAG-Agent\Test\Dataset\News_Category_Dataset_v3.json\News_Category_Dataset_v3.json"
    print(f"Loading dataset from: {json_path}")
    
    if not os.path.exists(json_path):
        print(f"Error: Dataset file not found at {json_path}")
        return
        
    documents = process_news_data(json_path, limit=20)
    
    print("Splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, 
        chunk_overlap=400,  
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]  
    )
    
    splits = text_splitter.split_documents(documents)
    
    print("Creating and persisting vector store...")
    persist_dir = os.path.join(os.path.dirname(__file__), "chroma_db")
    print(f"Vector store will be saved to: {persist_dir}")
    
    try:
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=persist_dir
        )
        
        vectorstore.persist()
        print("\nVector database created and persisted successfully!")
        print(f"Total documents processed: {len(documents)}")
        print(f"Total chunks created: {len(splits)}")
        
        if os.path.exists(persist_dir):
            print(f"Vector store directory exists at: {persist_dir}")
        else:
            print("Warning: Vector store directory was not created")
            
    except Exception as e:
        print(f"Error creating vector store: {str(e)}")

if __name__ == "__main__":
    main() 