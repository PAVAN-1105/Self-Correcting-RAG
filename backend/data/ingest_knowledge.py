import wikipedia
import requests
import time
import sys
from bs4 import BeautifulSoup

API_URL = "http://localhost:8000/api/v1/index"

# 1. Wikipedia Topics
TOPICS = [
    # Cricket Topics (Basics to Advanced)
    "Cricket",
    "History of cricket",
    "Laws of cricket",
    "Glossary of cricket terms",
    "Batting (cricket)",
    "Bowling (cricket)",
    "Fielding (cricket)",
    "Twenty20",
    "Test cricket",
    "One Day International",
    "Cricket World Cup",
    
    # AI and ML Topics (Basics to Advanced)
    "Artificial intelligence",
    "Machine learning",
    "Deep learning",
    "Neural network",
    "Transformer (machine learning model)",
    "Large language model",
    "Natural language processing",
    "Computer vision",
    "Reinforcement learning",
    "Supervised learning",
    "Unsupervised learning",
    "Generative artificial intelligence",
    "Artificial neural network"
]

# 2. Specific URLs to Scrape and Ingest
URLS = [
    # Add any specific webpage URLs you want to ingest here
    # "https://en.wikipedia.org/wiki/Attention_Is_All_You_Need",
]

def push_to_api(text: str, source_label: str) -> int:
    """Helper to send extracted text to the indexing API."""
    response = requests.post(
        API_URL,
        json={
            "text": text,
            "source": source_label
        }
    )
    if response.status_code == 200:
        data = response.json()
        chunks = data.get("indexed_chunks", 0)
        print(f"  -> ✅ Successfully indexed {chunks} chunks for '{source_label}'")
        return chunks
    else:
        print(f"  -> ❌ Failed to index '{source_label}'. Status code: {response.status_code}")
        return 0

def ingest_data():
    total_chunks = 0
    
    # Step 1: Ingest Wikipedia Topics
    if TOPICS:
        print(f"\\n--- Starting ingestion of {len(TOPICS)} Wikipedia topics ---")
        for topic in TOPICS:
            try:
                print(f"Fetching Wikipedia page for: {topic}")
                page = wikipedia.page(topic, auto_suggest=False)
                content = page.content
                
                print(f"  -> Downloaded {len(content)} characters. Sending to indexing API...")
                total_chunks += push_to_api(content, f"Wikipedia: {topic}")
                    
            except wikipedia.exceptions.DisambiguationError as e:
                print(f"  -> Skipped '{topic}' due to disambiguation: {e.options[:3]}")
            except wikipedia.exceptions.PageError:
                print(f"  -> Page '{topic}' not found on Wikipedia.")
            except Exception as e:
                print(f"  -> Error processing '{topic}': {str(e)}")
                
            time.sleep(1) # Be polite to Wikipedia APIs
            
    # Step 2: Ingest External URLs
    if URLS:
        print(f"\\n--- Starting ingestion of {len(URLS)} Web URLs ---")
        for url in URLS:
            try:
                print(f"Fetching URL: {url}")
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                
                # Parse the HTML and extract text
                soup = BeautifulSoup(response.text, 'html.parser')
                for script in soup(["script", "style", "nav", "footer", "header"]):
                    script.extract()
                    
                text = soup.get_text(separator=' ', strip=True)
                print(f"  -> Extracted {len(text)} characters. Sending to indexing API...")
                
                total_chunks += push_to_api(text, url)
                
            except Exception as e:
                print(f"  -> Error scraping or indexing URL: {e}")
                
            time.sleep(1)
            
    # Optional Step 3: Handle CLI argument for a single URL
    if len(sys.argv) > 1:
        single_url = sys.argv[1]
        print(f"\\n--- Ingesting single CLI URL: {single_url} ---")
        try:
            response = requests.get(single_url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.extract()
            text = soup.get_text(separator=' ', strip=True)
            print(f"  -> Extracted {len(text)} characters. Sending to indexing API...")
            total_chunks += push_to_api(text, single_url)
        except Exception as e:
            print(f"  -> Error scraping or indexing URL: {e}")

    print(f"\\n🎉 Ingestion Complete! Total semantic chunks indexed this session: {total_chunks}")

if __name__ == "__main__":
    ingest_data()
