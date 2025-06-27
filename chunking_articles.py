from dotenv import load_dotenv
import os
import json
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from uuid import uuid4
import shutil
import glob
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc

load_dotenv()


PROGRESS_FILE = "processing_progress.json"

def save_progress(processed_articles, total_chunks, failed_articles):
    progress = {
        "processed_count": processed_articles,
        "total_chunks": total_chunks,
        "failed_articles": failed_articles,
        "timestamp": time.time()
    }
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f)

def load_progress():
    try:
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {"processed_count": 0, "total_chunks": 0, "failed_articles": []}


embeddings = OllamaEmbeddings(
    model=os.getenv("EMBEDDING_MODEL"),
)


resume_mode = input("Resume from existing database? (y/n): ").lower() == 'y'

if not resume_mode and os.path.exists(os.getenv("DATABASE_LOCATION")):
    print("Deleting existing database...")
    shutil.rmtree(os.getenv("DATABASE_LOCATION"))
elif resume_mode:
    print("Resuming from existing database...")

vector_store = Chroma(
    collection_name=os.getenv("COLLECTION_NAME"),
    embedding_function=embeddings,
    persist_directory=os.getenv("DATABASE_LOCATION"), 
)


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800, # default to 1000
    chunk_overlap=160, # default to 200
    length_function=len,
    is_separator_regex=False,
)

def process_articles(articles_folder):
    extracted = []
    
    json_files = glob.glob(os.path.join(articles_folder, "*.json"))
    
    print(f"Found {len(json_files)} article files")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                article_data = json.load(f)
                
                title = article_data.get('title', os.path.splitext(os.path.basename(json_file))[0])
                content = article_data.get('content', '')
                
                source = f"osrs_wiki_{title.replace(' ', '_').replace('/', '_')}"
                
                content_parts = []
                
                if title:
                    content_parts.append(f"Title: {title}")
                
                if content:
                    cleaned_content = content.replace('\\n', '\n')
                    content_parts.append(cleaned_content.strip())
                
                raw_text = "\n\n".join(content_parts)
                
                if raw_text.strip():
                    extracted.append({
                        'title': title,
                        'raw_text': raw_text,
                        'source': source,
                        'file_path': json_file
                    })
                    
        except Exception as e:
            print(f"Error processing {json_file}: {str(e)}")
            continue
    
    return extracted

articles_folder = "datasets/clean_articles/"
articles = process_articles(articles_folder)

print(f"Successfully processed {len(articles)} articles")

progress = load_progress()
start_index = progress.get("processed_count", 0) if resume_mode else 0
total_chunks = progress.get("total_chunks", 0) if resume_mode else 0
failed_articles = progress.get("failed_articles", []) if resume_mode else []


if start_index > 0:
    print(f"Resuming from article {start_index + 1}")
    articles = articles[start_index:]

# Threading configuration (probably pointless as the vectorization cannot be threaded)
MAX_WORKERS = 6 
BATCH_SIZE = 500
processed_count = start_index
lock = threading.Lock()
start_time = time.time()

def process_article_batch(article_batch):
    batch_documents = []
    batch_ids = []
    batch_failed = []
    
    for article in article_batch:
        try:
            texts = text_splitter.create_documents(
                [article['raw_text']], 
                metadatas=[{
                    "source": article['source'], 
                    "title": article['title'],
                    "file_path": article['file_path'],
                    "article_length": len(article['raw_text'])
                }]
            )
            
            uuids = [str(uuid4()) for _ in range(len(texts))]
            batch_documents.extend(texts)
            batch_ids.extend(uuids)
            
        except Exception as e:
            batch_failed.append(f"{article['title']}: {str(e)}")
            continue
    
    return batch_documents, batch_ids, batch_failed

def update_progress(batch_num, total_batches, batch_size_actual, chunks_added):
    global processed_count, total_chunks
    
    with lock:
        processed_count += batch_size_actual
        total_chunks += chunks_added
        
        if batch_num % 5 == 0:
            elapsed = time.time() - start_time
            rate = processed_count / elapsed * 60 if elapsed > 0 else 0
            remaining = len(articles) - (processed_count - start_index)
            eta_minutes = remaining / (rate if rate > 0 else 1)
            
            print(f"Progress: {processed_count}/{len(articles) + start_index} ({processed_count/(len(articles) + start_index)*100:.1f}%)")
            print(f"Rate: {rate:.1f} articles/min, ETA: {eta_minutes:.0f} minutes")
            print(f"Total chunks: {total_chunks}")
            
            save_progress(processed_count, total_chunks, failed_articles)


article_batches = []
for i in range(0, len(articles), BATCH_SIZE):
    batch = articles[i:i + BATCH_SIZE]
    article_batches.append(batch)

print(f"Starting threaded processing of {len(articles)} articles...")
print(f"Using {MAX_WORKERS} threads with {len(article_batches)} batches")


with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    future_to_batch = {
        executor.submit(process_article_batch, batch): i 
        for i, batch in enumerate(article_batches)
    }
    
    for future in as_completed(future_to_batch):
        batch_num = future_to_batch[future]
        
        try:
            batch_documents, batch_ids, batch_failed = future.result()
            
            if batch_documents:
                print(f"Batch {batch_num + 1}: Adding {len(batch_documents)} chunks to vector store...")
                vector_store.add_documents(documents=batch_documents, ids=batch_ids)
                
                update_progress(batch_num, len(article_batches), len(article_batches[batch_num]), len(batch_documents))
            
            if batch_failed:
                with lock:
                    failed_articles.extend(batch_failed)
            
            del batch_documents, batch_ids
            
            if (batch_num + 1) % 10 == 0:
                gc.collect()
                
        except Exception as e:
            print(f"Error processing batch {batch_num + 1}: {str(e)}")

elapsed_time = (time.time() - start_time) / 3600
print(f"\nIngestion complete")
print(f"  Total articles processed: {processed_count}")
print(f"  Total chunks created: {total_chunks}")
print(f"  Failed articles: {len(failed_articles)}")
print(f"  Processing time: {elapsed_time:.1f} hours")

save_progress(processed_count, total_chunks, failed_articles)