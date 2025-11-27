import requests
import time
import os
import json
import re
from tqdm import tqdm

API_URL = "https://oldschool.runescape.wiki/api.php"
SAVE_DIR = "osrs_articles"

# Config
LIMIT = 500  
BATCH_SIZE = 50       # Reduced to be safer
MAX_RETRIES = 5       # How many times to retry a failed batch
BASE_WAIT = 2         # Seconds to wait before retrying
PAUSE_BETWEEN_PAGES = 0.5

HEADERS = {
    "User-Agent": "OSRS_Wiki_RAG_PROJECT/2.0 (Contact: @ludwilton"
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
    """
    Fetches a batch of titles with robust error handling for maxlag/rate limits.
    """
    titles_param = "|".join(titles)
    params = {
        "action": "query",
        "format": "json",
        "prop": "revisions",
        "rvprop": "content",
        "maxlag": 5,          # Ask server to fail fast if busy
        "titles": titles_param,
        "redirects": 1,
    }

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(API_URL, data=params, headers=HEADERS, timeout=30)
            
            # 1. Network/Server Errors
            if resp.status_code != 200:
                tqdm.write(f"⚠️  HTTP {resp.status_code}. Retrying in {BASE_WAIT}s...")
                time.sleep(BASE_WAIT * (attempt + 1))
                continue

            try:
                data = resp.json()
            except json.JSONDecodeError:
                tqdm.write("⚠️  Invalid JSON response. Retrying...")
                time.sleep(BASE_WAIT)
                continue

            # 2. MediaWiki API Errors (Maxlag, Rate Limit, etc.)
            if "error" in data:
                error_code = data["error"].get("code")
                error_info = data["error"].get("info", "Unknown error")
                
                # If server is busy (maxlag) or rate limited, we wait and retry
                if error_code in ["maxlag", "ratelimited", "readonly"]:
                    wait_time = int(resp.headers.get("Retry-After", BASE_WAIT * (attempt + 1)))
                    tqdm.write(f"⏳ API Busy ({error_code}). Waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    # Fatal error (e.g., bad params), skip this batch
                    tqdm.write(f"❌ API Error: {error_code} - {error_info}")
                    return {}

            # 3. Success
            results = {}
            if "query" in data:
                pages = data["query"].get("pages", {})
                for page_data in pages.values():
                    title = page_data.get("title")
                    if "missing" in page_data:
                        continue
                    if "revisions" in page_data and len(page_data["revisions"]) > 0:
                        content = page_data["revisions"][0].get("*", "")
                        results[title] = content
            
            return results

        except requests.exceptions.RequestException as e:
            tqdm.write(f"⚠️  Request failed: {e}. Retrying...")
            time.sleep(BASE_WAIT * (attempt + 1))

    tqdm.write(f"❌ Failed batch after {MAX_RETRIES} attempts.")
    return {}

def get_all_pages_list(limit=LIMIT):
    all_pages = []
    apcontinue = None
    print("Fetching list of all pages from Wiki API...")

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

            try:
                resp = requests.get(API_URL, params=params, headers=HEADERS, timeout=30)
                data = resp.json()
                
                if "error" in data:
                    tqdm.write(f"Error fetching list: {data['error'].get('info')}")
                    time.sleep(5)
                    continue

                pages = data.get("query", {}).get("allpages", [])
                if not pages:
                    break

                pbar.update(len(pages))
                for page in pages:
                    all_pages.append(page["title"])

                if "continue" in data and "apcontinue" in data["continue"]:
                    apcontinue = data["continue"]["apcontinue"]
                    time.sleep(PAUSE_BETWEEN_PAGES)
                else:
                    break
            except Exception as e:
                tqdm.write(f"Error reading page list: {e}")
                time.sleep(2)

    print(f"\nFinished fetching page list. Total: {len(all_pages)}")
    return all_pages

def main():
    try:
        all_wiki_pages = get_all_pages_list(limit=LIMIT)
    except KeyboardInterrupt:
        print("\nStopping...")
        return
    
    existing_files = set(os.listdir(SAVE_DIR))
    print(f"Pages currently in local folder: {len(existing_files)}")

    pages_to_fetch = []
    for title in all_wiki_pages:
        expected_filename = get_safe_filename(title)
        if expected_filename not in existing_files:
            pages_to_fetch.append(title)
            
    print(f"New pages to fetch: {len(pages_to_fetch)}")
    
    if not pages_to_fetch:
        print("Everything is up to date!")
        return

    with tqdm(total=len(pages_to_fetch), desc="Downloading Articles", unit="article") as pbar:
        for i in range(0, len(pages_to_fetch), BATCH_SIZE):
            batch = pages_to_fetch[i:i+BATCH_SIZE]
            
            try:
                wikitexts = get_wikitexts_batch_with_retry(batch)
                
                for title, content in wikitexts.items():
                    save_article(title, content)
                
            except Exception as e:
                tqdm.write(f"Critical error on batch {i}: {e}")
            
            pbar.update(len(batch))
            
            # polite delay
            time.sleep(0.5)
    
    print(f"\nFinished processing.")

if __name__ == "__main__":
    main()