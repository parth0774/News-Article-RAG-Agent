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
from typing import Dict, List

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

def load_news_data(file_path: str) -> List[Dict]:
    with open(file_path, 'r', encoding='utf-8') as f:
        articles = []
        for line in f:
            article = json.loads(line)
            articles.append(article)
    return articles

def process_news_data(articles: List[Dict]) -> List[Document]:
    documents = []
    for article in articles:
        # Combine headline and short_description for better context
        content = f"{article['headline']}\n\n{article['short_description']}"
        
        # Create document with all metadata fields
        doc = Document(
            page_content=content,
            metadata={
                "headline": article['headline'],
                "category": article['category'],
                "date": article['date'],
                "link": article['link'],
                "authors": article['authors'],
                "short_description": article['short_description']
            }
        )
        documents.append(doc)
    return documents

def create_vector_store(documents: List[Document], persist_dir: str):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    vectorstore.persist()
    return vectorstore

def main():
    print("Loading news data...")
    articles = load_news_data("News_Category_Dataset_v3.json")
    print(f"Loaded {len(articles)} articles")
    print("Processing articles...")
    documents = process_news_data(articles)
    print(f"Processed {len(documents)} documents")
    
    print("Creating vector store...")
    persist_dir = os.path.join(os.path.dirname(__file__), "chroma_db")
    vectorstore = create_vector_store(documents, persist_dir)
    print(f"Vector store created and persisted to {persist_dir}")
    
    print("\nVerifying vector store contents...")
    collection = vectorstore._collection
    docs = collection.get()
    print(f"Total documents in vector store: {len(docs['ids'])}")
    print("\nSample document metadata:")
    if docs['metadatas']:
        sample_metadata = docs['metadatas'][0]
        for key, value in sample_metadata.items():
            print(f"{key}: {value}")

if __name__ == "__main__":
    main() 