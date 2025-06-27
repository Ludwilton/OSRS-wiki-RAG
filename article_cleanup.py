import json
import os
import re

def clean_with_regex(wikitext):
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
    text = re.sub(r'^=+\s*(.+?)\s*=+$', r'\1', text, flags=re.MULTILINE)
    
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
        # Keep non-empty lines and meaningful content
        if line and not re.match(r'^\s*[{}|!=]+\s*$', line):
            cleaned_lines.append(line)
        elif not cleaned_lines or cleaned_lines[-1]:  # Add empty line only if previous wasn't empty
            cleaned_lines.append('')
    
    # Join and final whitespace cleanup
    text = '\n'.join(cleaned_lines)
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Max 2 consecutive newlines
    text = re.sub(r'[ \t]+', ' ', text)             # Multiple spaces -> single
    
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

def process_all_files(input_dir, output_dir):
    """
    Process all JSON files in the input directory.
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return
    
    print(f"Found {len(json_files)} JSON files to process")

    
    successful = 0
    failed = 0
    
    for i, filename in enumerate(json_files, 1):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        print(f"\n[{i}/{len(json_files)}] Processing: {filename}")
        
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if 'wikitext' not in data:
                print(f"No 'wikitext' field found, skipping")
                continue
            
            original = data['wikitext']
            cleaned = clean_with_regex(original)
            
            result = {
                'title': data.get('title', ''),
                'content': cleaned
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            reduction = ((len(original) - len(cleaned)) / len(original) * 100) if len(original) > 0 else 0
            print(f" Success - Reduction: {reduction:.1f}% ({len(original)} → {len(cleaned)} chars)")
            successful += 1
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            failed += 1
    
    print(f"\nProcessing complete")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Output directory: {output_dir}")

if __name__ == "__main__":
    input_directory = "datasets/osrs_articles"
    output_directory = "datasets/clean_articles"
    
    process_all_files(input_directory, output_directory)