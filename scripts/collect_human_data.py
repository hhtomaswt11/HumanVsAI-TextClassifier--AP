import requests
import csv
import time
import os

def limit_word_count(text, min_words=80, max_words=120):
    if not text:
        return None
    
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    accumulated_words = []
    current_count = 0
    
    for sentence in sentences:
        sentence_words = sentence.split()
        if not sentence_words:
            continue
            
        if current_count + len(sentence_words) <= max_words:
            accumulated_words.extend(sentence_words)
            current_count += len(sentence_words)
            if current_count >= min_words:
                break
        else:
            if current_count >= min_words:
                break
            
            needed = max_words - current_count
            truncated_sentence = sentence_words[:needed]
            accumulated_words.extend(truncated_sentence)
            current_count += len(truncated_sentence)
            break
            
    if current_count >= min_words:
        result = " ".join(accumulated_words)
        if not result.endswith(('.', '!', '?')):
            result += "..."
        return result
        
    return None

import random

def fetch_wikipedia_content(topic, vary_sections=True, is_retry=False):
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "titles": topic,
        "prop": "extracts",
        "explaintext": 1,
        "redirects": 1
    }
    headers = {
        'User-Agent': 'DatasetGenerator/1.0 (contact: your-email@example.com)'
    }
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            pages = data.get('query', {}).get('pages', {})
            page_data = next(iter(pages.values()))
            
            if "-1" in pages: # Not found
                if not is_retry:
                    search_params = {
                        "action": "query",
                        "list": "search",
                        "srsearch": topic,
                        "format": "json"
                    }
                    s_res = requests.get(url, params=search_params, headers=headers, timeout=10)
                    if s_res.status_code == 200:
                        results = s_res.json().get('query', {}).get('search', [])
                        if results:
                            return fetch_wikipedia_content(results[0]['title'], vary_sections, is_retry=True)
                return None
            
            full_text = page_data.get('extract', "")
            if not full_text:
                return None
            
            import re
            parts = re.split(r'\n==+ (.+?) ==+\n', full_text)
            
            sections = []
            if parts[0].strip():
                sections.append(("Introduction", parts[0].strip()))
            
            for i in range(1, len(parts), 2):
                header = parts[i].strip().lower()
                content = parts[i+1].strip()
                exclude = ["references", "see also", "external links", "further reading", "notes", "citations", "bibliography"]
                if not any(ex in header for ex in exclude) and len(content.split()) > 10:
                    sections.append((header, content))
            
            if not sections:
                return None
            
            if vary_sections and len(sections) > 1:
                idx = random.randint(0, len(sections) - 1)
                all_text = " ".join([s[1] for s in sections[idx:]])
                return all_text
            else:
                return " ".join([s[1] for s in sections])
                
        return None
    except Exception as e:
        print(f"Error fetching {topic}: {e}")
        return None


def search_wikipedia_candidates(topic, max_results=5):
    """Return a list of candidate page titles from Wikipedia search for a topic."""
    url = "https://en.wikipedia.org/w/api.php"
    headers = {
        'User-Agent': 'DatasetGenerator/1.0 (contact: your-email@example.com)'
    }
    params = {
        "action": "query",
        "list": "search",
        "srsearch": topic,
        "srlimit": max_results,
        "format": "json"
    }
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        if resp.status_code == 200:
            results = resp.json().get('query', {}).get('search', [])
            return [r['title'] for r in results]
    except Exception as e:
        print(f"Search error for {topic}: {e}")
    return []

def main():
    input_file = "temas_novos.txt"
    output_file = "dataset_human1.csv"
    MAX_RETRIES = 5
    MAX_CANDIDATES = 5
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    with open(input_file, 'r') as f:
        topics = [line.strip() for line in f if line.strip()]

    print(f"Found {len(topics)} topics. Starting fetch...")

    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['text', 'label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i, topic in enumerate(topics):
            if i > 0 and i % 10 == 0:
                print(f"Processed {i}/{len(topics)}...")
                time.sleep(1) 

            success = False
            for attempt in range(MAX_RETRIES):
                vary = ((i + attempt) % 2 == 0)
                raw_content = fetch_wikipedia_content(topic, vary_sections=vary)
                summary = limit_word_count(raw_content)
                if summary:
                    writer.writerow({
                        'text': summary,
                        'label': 'Human',
                    })
                    success = True
                    break

            if not success:
                candidates = search_wikipedia_candidates(topic, max_results=MAX_CANDIDATES)
                for cand in candidates:
                    raw_content = fetch_wikipedia_content(cand, vary_sections=True)
                    summary = limit_word_count(raw_content)
                    if summary:
                        writer.writerow({
                            'text': summary,
                            'label': 'Human',
                        })
                        success = True
                        break

            if not success:
                word_count = len(raw_content.split()) if raw_content else 0
                print(f"Warning: Could not get 80-120 words for '{topic}' (Got {word_count})")

    print(f"Finished! Human dataset saved to {output_file}")

if __name__ == "__main__":
    main()
