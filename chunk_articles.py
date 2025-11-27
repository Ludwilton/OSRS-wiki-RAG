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
from typing import List, Tuple
from tqdm import tqdm

def load_environment():
    if load_dotenv(".env", override=True):
        print("[env] Loaded environment from .env")
    else:
        print("[env] Warning: No .env file found")

load_environment()


PROGRESS_FILE = "processing_progress.json"
FAILED_MANIFEST = "failed_articles.txt"

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


def _require_env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return val


EMBEDDING_MODEL = _require_env("EMBEDDING_MODEL")
COLLECTION_NAME = _require_env("COLLECTION_NAME")
DATABASE_LOCATION = _require_env("DATABASE_LOCATION")
ARTICLES_DIR = os.getenv("CHUNK_INPUT_DIR", "clean_articles")
RESUME_MODE = os.getenv("CHUNK_RESUME", "0") == "1"
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
if CHUNK_OVERLAP >= CHUNK_SIZE:
    adj = max(1, int(CHUNK_SIZE * 0.15))
    print(f"[config] overlap {CHUNK_OVERLAP} >= size {CHUNK_SIZE}; adjusted overlap -> {adj}")
    CHUNK_OVERLAP = adj
MAX_WORKERS = int(os.getenv("CHUNK_MAX_WORKERS", "4"))
BATCH_SIZE = int(os.getenv("CHUNK_BATCH_SIZE", "400"))
PERSIST_EVERY = int(os.getenv("CHUNK_PERSIST_EVERY", "5"))
PROGRESS_EVERY = int(os.getenv("CHUNK_PROGRESS_EVERY", "1"))
RETRY_ADD_DOCS = int(os.getenv("CHUNK_RETRY_ADD_DOCS", "2"))

# CHANGE THIS LINE
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

if not RESUME_MODE and os.path.exists(DATABASE_LOCATION):
    print("Deleting existing database (set CHUNK_RESUME=1 to append)...")
    shutil.rmtree(DATABASE_LOCATION)
elif RESUME_MODE:
    print("Resuming from existing database...")

vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=DATABASE_LOCATION,
)

def persist_vector_store():
    """Safely persist the vector store if the backend supports it."""
    try:
        if 'persist' in dir(vector_store):
            vector_store.persist()
        elif hasattr(vector_store, '_client') and 'persist' in dir(vector_store._client):
            vector_store._client.persist()
        else:
            pass
    except Exception as e:
        tqdm.write(f"[persist] failed: {e}")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    is_separator_regex=False,
)


def process_articles(articles_folder: str) -> List[dict]:
    extracted: List[dict] = []
    json_files = sorted(glob.glob(os.path.join(articles_folder, "*.json")))
    print(f"Found {len(json_files)} article files in {articles_folder}")
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                article_data = json.load(f)
            title = article_data.get('title', os.path.splitext(os.path.basename(json_file))[0])
            content = article_data.get('content') or ''
            article_id = str(uuid4())
            source = f"https://oldschool.runescape.wiki/w/{title.replace(' ', '_').replace('/', '_')}"
            
            parts = [f"Title: {title}"]
            if content.strip():
                parts.append(content.strip())
            raw_text = "\n\n".join(parts)
            
            if raw_text.strip():
                extracted.append({
                    'title': title,
                    'raw_text': raw_text,
                    'source': source,
                    'file_path': json_file,
                    'article_id': article_id,
                })
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
    return extracted


articles = process_articles(ARTICLES_DIR)

print(f"Discovered {len(articles)} articles with content")

progress = load_progress() if RESUME_MODE else {"processed_count": 0, "total_chunks": 0, "failed_articles": []}
start_index = progress.get("processed_count", 0) if RESUME_MODE else 0
total_chunks = progress.get("total_chunks", 0) if RESUME_MODE else 0
failed_articles = progress.get("failed_articles", []) if RESUME_MODE else []

total_article_count = len(articles)
if start_index > 0:
    if start_index >= total_article_count:
        print("Progress file indicates all articles processed. Nothing to do.")
    else:
        print(f"Resuming from article index {start_index} (1-based {start_index + 1})")
        articles = articles[start_index:]

processed_count = start_index
lock = threading.Lock()
start_time = time.time()

def process_article_batch(article_batch: List[dict]) -> Tuple[List, List[str], List[str]]:
    batch_documents = []
    batch_ids: List[str] = []
    batch_failed: List[str] = []
    for article in article_batch:
        try:
            texts = text_splitter.create_documents(
                [article['raw_text']],
                metadatas=[{
                    "source": article['source'],
                    "title": article['title'],
                    "file_path": article['file_path'],
                    "article_length": len(article['raw_text']),
                    "parent_id": article['article_id'],
                }]
            )
            uuids = [str(uuid4()) for _ in range(len(texts))]
            batch_documents.extend(texts)
            batch_ids.extend(uuids)
        except Exception as e:
            batch_failed.append(f"{article['title']}: {e}")
    return batch_documents, batch_ids, batch_failed


def update_progress(batch_index: int, batch_size_actual: int, chunks_added: int, total_batches: int):
    global processed_count, total_chunks
    with lock:
        processed_count += batch_size_actual
        total_chunks += chunks_added
        
        if (batch_index + 1) % PROGRESS_EVERY == 0:
            save_progress(processed_count, total_chunks, failed_articles)


article_batches: List[List[dict]] = [articles[i:i + BATCH_SIZE] for i in range(0, len(articles), BATCH_SIZE)]
print(f"Starting processing: {len(articles)} remaining articles -> {len(article_batches)} batches | chunk_size={CHUNK_SIZE} overlap={CHUNK_OVERLAP}")
print(f"Workers={MAX_WORKERS} persist_every={PERSIST_EVERY}")


with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    future_to_batch = {
        executor.submit(process_article_batch, batch): i 
        for i, batch in enumerate(article_batches)
    }
    
    with tqdm(total=len(article_batches), desc="Ingesting Batches", unit="batch") as pbar:
        for future in as_completed(future_to_batch):
            batch_num = future_to_batch[future]
            try:
                batch_documents, batch_ids, batch_failed = future.result()
                if batch_documents:
                    for attempt in range(1, RETRY_ADD_DOCS + 2):
                        try:
                            vector_store.add_documents(documents=batch_documents, ids=batch_ids)
                            break
                        except Exception as add_e:
                            if attempt > RETRY_ADD_DOCS:
                                raise
                            backoff = 2 ** (attempt - 1)
                            tqdm.write(f"  add_documents failed (attempt {attempt}): {add_e} -> retry in {backoff}s")
                            time.sleep(backoff)
                    
                    update_progress(batch_num, len(article_batches[batch_num]), len(batch_documents), len(article_batches))
                    
                    if (batch_num + 1) % PERSIST_EVERY == 0:
                        persist_vector_store()
                
                if batch_failed:
                    with lock:
                        failed_articles.extend(batch_failed)
                
                del batch_documents, batch_ids
                if (batch_num + 1) % 10 == 0:
                    gc.collect()
            
            except Exception as e:
                tqdm.write(f"Error processing batch {batch_num + 1}: {e}")
            
            if failed_articles:
                try:
                    with open(FAILED_MANIFEST, 'w', encoding='utf-8') as fm:
                        fm.write("\n".join(failed_articles))
                except Exception:
                    pass
            
            pbar.update(1)
            pbar.set_postfix({"Chunks": total_chunks, "Failed": len(failed_articles)})


elapsed_time = (time.time() - start_time) / 3600
print("\nIngestion complete")
print(f"  Total articles processed: {processed_count}")
print(f"  Total chunks created: {total_chunks}")
print(f"  Failed articles: {len(failed_articles)}")
print(f"  Processing time: {elapsed_time:.1f} hours")
if failed_articles:
    print(f"  Failed manifest written to {FAILED_MANIFEST}")
avg_chunks = (total_chunks / processed_count) if processed_count else 0
print(f"  Avg chunks/article: {avg_chunks:.2f}")
persist_vector_store()

save_progress(processed_count, total_chunks, failed_articles)