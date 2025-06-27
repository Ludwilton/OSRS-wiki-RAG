import requests
import time
import os
import json
import re

API_URL = "https://oldschool.runescape.wiki/api.php"
SAVE_DIR = "osrs_articles"

LIMIT = 500  
BATCH_SIZE = 50  
PAUSE_BETWEEN_BATCHES = 2  # necessary to follow wikimedia guidelines
PAUSE_BETWEEN_PAGES = 0.5  
os.makedirs(SAVE_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": "OSRS_DATA_PRJ_TEST 1.0 ludwilton"
}

def is_redirect(title):
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "info",
        "redirects": ""
    }
    resp = requests.get(API_URL, params=params, headers=HEADERS)
    data = resp.json()
    page = list(data["query"]["pages"].values())[0]
    return "redirect" in page

def get_wikitext(title):
    params = {
        "action": "parse",
        "format": "json",
        "page": title,
        "prop": "wikitext"
    }
    resp = requests.get(API_URL, params=params, headers=HEADERS)
    data = resp.json()
    return data["parse"]["wikitext"]["*"]


def save_article(title, content):
    safe_title = re.sub(r'[<>:"/\\|?*]', '_', title)
    safe_title = safe_title.strip()  
    path = os.path.join(SAVE_DIR, f"{safe_title}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"title": title, "wikitext": content}, f, ensure_ascii=False, indent=2)

def get_wikitexts_batch(titles, batch_size=20):
    all_results = {}
    
    for i in range(0, len(titles), batch_size):
        batch = titles[i:i+batch_size]
        titles_param = "|".join(batch)
        
        params = {
            "action": "parse",
            "format": "json",
            "prop": "wikitext",
            "maxlag": 5,
            "page": batch[0],
            "redirects": 1,
        }
        
        if len(batch) > 1:
            params = {
                "action": "query",
                "format": "json",
                "prop": "revisions",
                "rvprop": "content",
                "maxlag": 5,
                "titles": titles_param,
                "redirects": 1,
            }
        
        print(f"Fetching batch of {len(batch)} pages")
        resp = requests.get(API_URL, params=params, headers=HEADERS)
        
        try:
            data = resp.json()
            
            if "query" in data:
                pages = data["query"].get("pages", {})
                for page_id, page_data in pages.items():
                    title = page_data.get("title")
                    if "revisions" in page_data and len(page_data["revisions"]) > 0:
                        content = page_data["revisions"][0].get("*", "")
                        all_results[title] = content
            elif "parse" in data:
                title = data["parse"]["title"]
                content = data["parse"]["wikitext"]["*"]
                all_results[title] = content
        except Exception as e:
            print(f"Error fetching batch: {e}")
        
        time.sleep(1)
    
    return all_results



def get_all_pages(limit=LIMIT):
    all_pages = []
    apcontinue = None
    total_pages = 0

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

        print(f"Fetching pages with apcontinue={apcontinue}...")

        resp = requests.get(API_URL, params=params, headers=HEADERS)
        try:
            data = resp.json()
        except Exception as e:
            print("Failed reading JSON.")
            print(f"Status code: {resp.status_code}")
            print(f"Error: {e}")
            print(resp.text[:500])
            break

        pages = data.get("query", {}).get("allpages", [])
        if not pages:
            print("No more pages found, exiting.")
            break

        total_pages += len(pages)
        print(f"Fetched {len(pages)} pages, total: {total_pages}")

        for page in pages:
            all_pages.append(page["title"])

        if "continue" in data and "apcontinue" in data["continue"]:
            apcontinue = data["continue"]["apcontinue"]
            time.sleep(PAUSE_BETWEEN_PAGES)
        else:
            break

    return all_pages

def save_progress(processed_titles):
    with open("progress.json", "w", encoding="utf-8") as f:
        json.dump(processed_titles, f)

def load_progress():
    try:
        with open("progress.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def main():
    global HEADERS
    HEADERS = {
        "User-Agent": "OSRS_WIKIRAG_PRJ 1.0 ludwilton"
    }
    
    processed_titles = load_progress()
    print(f"Found {len(processed_titles)} previously processed titles.")
    
    pages = get_all_pages(limit=LIMIT)
    print(f"\nTotal found pages: {len(pages)}")
    
    pages = [p for p in pages if p not in processed_titles]
    print(f"Remaning pages to fetch: {len(pages)}")
    
    for i in range(0, len(pages), BATCH_SIZE):
        batch = pages[i:i+BATCH_SIZE]
        print(f"\nFetching batch {i//BATCH_SIZE + 1}/{len(pages)//BATCH_SIZE + 1} ({len(batch)} pages)")
        
        try:
            wikitexts = get_wikitexts_batch(batch, batch_size=BATCH_SIZE)
            
            for title, content in wikitexts.items():
                save_article(title, content)
                processed_titles.append(title)
            save_progress(processed_titles)
            
        except Exception as e:
            print(f"Error on batch {i//BATCH_SIZE + 1}: {e}")
        
        # pause necessary to follow wikimedia API guidelines
        print(f"Pausing for {PAUSE_BETWEEN_BATCHES} seconds...")
        time.sleep(PAUSE_BETWEEN_BATCHES)
    
    print(f"\n Finished, {len(processed_titles)} articles fetched.")

if __name__ == "__main__":
    main()
