import os, json, textwrap, random
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

load_dotenv()

COLLECTION_NAME = os.getenv("COLLECTION_NAME")
DATABASE_LOCATION = os.getenv("DATABASE_LOCATION")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=DATABASE_LOCATION
)

col = store._collection

count = col.count()
print(f"Total documents (chunks): {count}")

N = 5
sample = col.get(include=["metadatas","documents"], limit=N)
for i, (doc, meta, id_) in enumerate(zip(sample["documents"], sample["metadatas"], sample["ids"])):
    print("="*70)
    print(f"Sample #{i}  ID={id_}")
    print(f"Source: {meta.get('source')}  Title: {meta.get('title')}  Article length: {meta.get('article_length')}")
    print("Preview:")
    print(textwrap.shorten(doc.replace("\n"," \\n "), width=400, placeholder=" ..."))

only_title = 0
short_docs = 0
BATCH = 1000
for offset in range(0, count, BATCH):
    batch = col.get(include=["documents"], limit=BATCH, offset=offset)
    for d in batch["documents"]:
        lines = [l for l in d.splitlines() if l.strip()]
        if len(lines) <= 1:
            only_title += 1
        if len(d) < 80:
            short_docs += 1

print("\nStats:")
print(f"  Chunks with only one non-empty line: {only_title}")
print(f"  Very short chunks (<80 chars): {short_docs}")

if only_title:
    print("\nExamples of 'only title' docs:")
    batch = col.get(include=["documents","metadatas"], limit=200)
    shown = 0
    for d, m, id_ in zip(batch["documents"], batch["metadatas"], batch["ids"]):
        lines = [l for l in d.splitlines() if l.strip()]
        if len(lines) <= 1:
            print(f"- ID {id_} Source={m.get('source')} Doc={repr(d)}")
            shown += 1
        if shown >= 5:
            break

# Random sample helper (uncomment to inspect random docs)
# rand_ids = random.sample(sample["ids"], min(3, len(sample["ids"])))
# print("Random IDs:", rand_ids)