import json
import os
import re
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


def clean_with_regex(wikitext):
    """cleans the wiki articles with regex. super messy and removes quite a bunch of important tables. needs rework."""
    text = wikitext
    
    # templates
    text = remove_nested_templates(text)
    
    # HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # wikilinks
    text = re.sub(r'\[\[(?:File|Image|Category):[^\]]+\]\]', '', text)
    text = re.sub(r'\[\[([^\]|]+)\|([^\]]+)\]\]', r'\2', text)
    text = re.sub(r'\[\[([^\]]+)\]\]', r'\1', text)
    text = re.sub(r'\[([^\]]+)\]', '', text)
    
    # formatting
    text = re.sub(r"'''([^']+)'''", r'\1', text)
    text = re.sub(r"''([^']+)''", r'\1', text)
    text = re.sub(r'^=+\s*(.+?)\s*=+$', r'## \1', text, flags=re.MULTILINE)
    
    # tables
    text = re.sub(r'^\s*\{\|.*?\|\}\s*$', '', text, flags=re.MULTILINE | re.DOTALL)
    text = re.sub(r'^\s*[|!].*$', '', text, flags=re.MULTILINE)
    
    # comments
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)

    text = re.sub(r"'''([^']+)'''", r'\1', text)  # Bold
    text = re.sub(r"''([^']+)''", r'\1', text)    # Italic
    
    # Clean up section headers
    text = re.sub(r'^=+\s*(.+?)\s*=+\s*$', r'\1', text, flags=re.MULTILINE)
    
    # Remove any remaining table markup
    text = re.sub(r'^\s*\{\|.*?\|\}\s*$', '', text, flags=re.MULTILINE | re.DOTALL)
    text = re.sub(r'^\s*[|!].*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\|-.*$', '', text, flags=re.MULTILINE)
    
    # Clean HTML entities
    text = re.sub(r'&nbsp;', ' ', text)
    text = re.sub(r'&[a-zA-Z0-9#]+;', '', text)
    
    # Remove empty lines and excessive whitespace
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        if line and not re.match(r'^\s*[{}|!=]+\s*$', line):
            cleaned_lines.append(line)
        elif not cleaned_lines or cleaned_lines[-1]:
            cleaned_lines.append('')
    
    # Join and final whitespace cleanup
    text = '\n'.join(cleaned_lines)
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    
    return text.strip()


def remove_nested_templates(text):
    """
    Remove nested {{ ... }} templates.
    """
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

def process_file_wrapper(args):
    """Wrapper function to unpack arguments for multiprocessing"""
    return process_one_file(*args)

def process_one_file(filename, input_dir, output_dir):
    """
    Worker function to process a single file. 
    Must be top-level or picklable for multiprocessing.
    """
    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, filename)
    
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'wikitext' not in data:
            return False, f"Skipping {filename}: No 'wikitext' field"
        
        original = data['wikitext']
        cleaned = clean_with_regex(original)
        
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
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    all_input_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    
    if not all_input_files:
        print(f"No JSON files found in {input_dir}")
        return

    existing_output_files = set(os.listdir(output_dir))
    files_to_process = [f for f in all_input_files if f not in existing_output_files]

    print(f"Total input files found: {len(all_input_files)}")
    print(f"Already processed:       {len(all_input_files) - len(files_to_process)}")
    print(f"Remaining to process:    {len(files_to_process)}")
    
    if not files_to_process:
        print("Nothing new to process.")
        return

    successful = 0
    failed = 0
    
    max_workers = multiprocessing.cpu_count()
    print(f"Starting multiprocessing with {max_workers} workers...")
    
    tasks = [(f, input_dir, output_dir) for f in files_to_process]
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_file_wrapper, task) for task in tasks]
        
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

    print(f"\nProcessing complete")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Output directory: {output_dir}")

if __name__ == "__main__":
    input_directory = "osrs_articles"
    output_directory = "clean_articles"
    
    process_all_files(input_directory, output_directory)