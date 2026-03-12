import json
import os
import re
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


def clean_with_regex(wikitext):
    """very messy parser, however retrieval quality is fine with this."""
    text = wikitext
    text = remove_nested_templates(text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\[\[(?:File|Image|Category):[^\]]+\]\]', '', text)
    text = re.sub(r'\[\[([^\]|]+)\|([^\]]+)\]\]', r'\2', text)
    text = re.sub(r'\[\[([^\]]+)\]\]', r'\1', text)
    text = re.sub(r'\[([^\]]+)\]', '', text)
    text = re.sub(r"'''([^']+)'''", r'\1', text)
    text = re.sub(r"''([^']+)''", r'\1', text)
    text = re.sub(r'^\s*\{\|.*?\|\}\s*$', '', text, flags=re.MULTILINE | re.DOTALL)
    text = re.sub(r'^\s*[|!].*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\|-.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    text = re.sub(r'^=+\s*(.+?)\s*=+\s*$', r'\1', text, flags=re.MULTILINE)
    text = re.sub(r'&nbsp;', ' ', text)
    text = re.sub(r'&[a-zA-Z0-9#]+;', '', text)

    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        if line and not re.match(r'^\s*[{}|!=]+\s*$', line):
            cleaned_lines.append(line)
        elif not cleaned_lines or cleaned_lines[-1]:
            cleaned_lines.append('')
            
    text = '\n'.join(cleaned_lines)
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    
    return text.strip()


def remove_nested_templates(text):
    result = []
    i = 0
    
    while i < len(text):
        if i < len(text) - 1 and text[i:i+2] == '{{':
            brace_count = 1
            j = i + 2
            
            while j < len(text) - 1 and brace_count > 0:
                if text[j:j+2] == '{{':
                    brace_count += 1
                    j += 2
                elif text[j:j+2] == '}}':
                    brace_count -= 1
                    j += 2
                else:
                    j += 1
            
            i = j if brace_count == 0 else i + 1
        else:
            result.append(text[i])
            i += 1
    
    return ''.join(result)


def process_one_file(filename, input_dir, output_dir):
    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, filename)
    
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'wikitext' not in data:
            return False, f"Skipping {filename}: No 'wikitext' field"
        
        cleaned = clean_with_regex(data['wikitext'])
        
        result = {
            'title': data.get('title', ''),
            'content': cleaned
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
            
        return True, None
        
    except Exception as e:
        return False, f"Error processing {filename}: {e}"


def process_all_files(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    all_input_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    if not all_input_files:
        return

    files_to_process = [f for f in all_input_files if f not in set(os.listdir(output_dir))]
    if not files_to_process:
        print("Nothing new to process.")
        return

    print(f"{len(files_to_process)} files to process ({len(all_input_files) - len(files_to_process)} already done)")

    successful = 0
    failed = 0
    max_workers = multiprocessing.cpu_count()
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_one_file, f, input_dir, output_dir) for f in files_to_process]
        
        with tqdm(total=len(files_to_process), desc="Cleaning Articles", unit="file") as pbar:
            for future in as_completed(futures):
                success, error_msg = future.result()
                
                if success:
                    successful += 1
                else:
                    failed += 1
                    if error_msg:
                        tqdm.write(error_msg)
                
                pbar.update(1)

    print(f"Done — {successful} succeeded, {failed} failed")

if __name__ == "__main__":
    input_directory = "osrs_articles"
    output_directory = "clean_articles"
    process_all_files(input_directory, output_directory)