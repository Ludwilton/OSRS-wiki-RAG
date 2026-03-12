import requests
import time
import os
import json
import re
from tqdm import tqdm

API_URL = "https://oldschool.runescape.wiki/api.php"
SAVE_DIR = "osrs_articles"


LIMIT = 500  
BATCH_SIZE = 50
MAX_RETRIES = 2
PAUSE_BETWEEN_PAGES = 0.5

HEADERS = {
    "User-Agent": "OSRS_Wiki_RAG_PROJECT/2.0 (Contact: @ludwilton)"
}

os.makedirs(SAVE_DIR, exist_ok=True)

def get_safe_filename(title):
    safe_title = re.sub(r'[<>:"/\\|?*]', '_', title).strip()
    return f"{safe_title}.json"

def save_article(title, content):
    filename = get_safe_filename(title)
    path = os.path.join(SAVE_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"title": title, "wikitext": content}, f, ensure_ascii=False, indent=2)

def get_wikitexts_batch_with_retry(titles):
    titles_param = "|".join(titles)
    params = {
        "action": "query",
        "format": "json",
        "prop": "revisions",
        "rvprop": "content",
        "maxlag": 5,
        "titles": titles_param,
        "redirects": 1,
    }

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(API_URL, data=params, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            
            if "error" in data:
                tqdm.write(f"API error: {data['error'].get('code')}")
                time.sleep(2)
                continue

            results = {}
            pages = data.get("query", {}).get("pages", {})
            for page_data in pages.values():
                if "missing" not in page_data and "revisions" in page_data:
                    title = page_data.get("title")
                    content = page_data["revisions"][0].get("*", "")
                    results[title] = content
            
            return results

        except requests.RequestException:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2)
                continue
    
    return {}

def get_all_pages_list(limit=LIMIT):
    all_pages = []
    apcontinue = None

    with tqdm(desc="Fetching Titles", unit=" titles") as pbar:
        while True:
            params = {
                "action": "query",
                "format": "json",
                "list": "allpages",
                "aplimit": limit,
                "apnamespace": 0,
                "apfilterredir": "nonredirects", 
            }
            if apcontinue:
                params["apcontinue"] = apcontinue

            resp = requests.get(API_URL, params=params, headers=HEADERS, timeout=30)
            data = resp.json()
            
            pages = data.get("query", {}).get("allpages", [])
            if not pages:
                break

            pbar.update(len(pages))
            all_pages.extend(page["title"] for page in pages)

            if "continue" in data and "apcontinue" in data["continue"]:
                apcontinue = data["continue"]["apcontinue"]
                time.sleep(PAUSE_BETWEEN_PAGES)
            else:
                break

    return all_pages

def main():
    try:
        all_wiki_pages = get_all_pages_list(limit=LIMIT)
    except KeyboardInterrupt:
        print("\nInterrupted")
        return
    
    existing_files = set(os.listdir(SAVE_DIR))
    pages_to_fetch = []
    for title in all_wiki_pages:
        expected_filename = get_safe_filename(title)
        if expected_filename not in existing_files:
            pages_to_fetch.append(title)
    
    if not pages_to_fetch:
        print("Up to date")
        return

    print(f"Fetching {len(pages_to_fetch)} articles")
    with tqdm(total=len(pages_to_fetch), desc="Downloading", unit="article") as pbar:
        for i in range(0, len(pages_to_fetch), BATCH_SIZE):
            batch = pages_to_fetch[i:i+BATCH_SIZE]
            wikitexts = get_wikitexts_batch_with_retry(batch)
            
            for title, content in wikitexts.items():
                save_article(title, content)
            
            pbar.update(len(batch))
            time.sleep(0.5)
    
    print("Done")

if __name__ == "__main__":
    main()