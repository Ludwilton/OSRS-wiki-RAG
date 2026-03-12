from dotenv import load_dotenv
import os
import json
import glob
import shutil
from uuid import uuid4
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List
from tqdm import tqdm
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import multiprocessing
load_dotenv()


def _require_env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return val


EMBEDDING_MODEL = _require_env("EMBEDDING_MODEL")
COLLECTION_NAME = _require_env("COLLECTION_NAME")
DATABASE_LOCATION = _require_env("DATABASE_LOCATION")
ARTICLES_DIR = "clean_articles"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
MAX_WORKERS = multiprocessing.cpu_count() - 1
BATCH_SIZE = 400

if CHUNK_OVERLAP >= CHUNK_SIZE:
    CHUNK_OVERLAP = max(1, int(CHUNK_SIZE * 0.15))

if os.path.exists(DATABASE_LOCATION):
    shutil.rmtree(DATABASE_LOCATION)

embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=DATABASE_LOCATION,
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    is_separator_regex=False,
)


def load_articles(folder: str) -> List[dict]:
    articles = []
    for path in sorted(glob.glob(os.path.join(folder, "*.json"))):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        title = data.get("title", os.path.splitext(os.path.basename(path))[0])
        content = data.get("content", "").strip()
        raw_text = f"Title: {title}\n\n{content}" if content else f"Title: {title}"
        articles.append({
            "title": title,
            "raw_text": raw_text,
            "source": f"https://oldschool.runescape.wiki/w/{title.replace(' ', '_').replace('/', '_')}",
            "file_path": path,
            "article_id": str(uuid4()),
        })
    return articles


def chunk_article(article: dict):
    docs = text_splitter.create_documents(
        [article["raw_text"]],
        metadatas=[{
            "source": article["source"],
            "title": article["title"],
            "file_path": article["file_path"],
            "article_length": len(article["raw_text"]),
            "parent_id": article["article_id"],
        }]
    )
    return docs, [str(uuid4()) for _ in docs]


articles = load_articles(ARTICLES_DIR)
print(f"Loaded {len(articles)} articles")

with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = [executor.submit(chunk_article, a) for a in articles]

    with tqdm(total=len(articles), desc="Chunking", unit="article") as pbar:
        all_docs, all_ids = [], []
        for future in as_completed(futures):
            docs, ids = future.result()
            all_docs.extend(docs)
            all_ids.extend(ids)
            pbar.update(1)

print(f"Ingesting {len(all_ids)} chunks...")
for i in range(0, len(all_docs), BATCH_SIZE):
    vector_store.add_documents(documents=all_docs[i:i + BATCH_SIZE], ids=all_ids[i:i + BATCH_SIZE])

print(f"Done — {len(all_ids)} chunks from {len(articles)} articles")