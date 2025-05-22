import requests
from bs4 import BeautifulSoup
import os
from whoosh.index import create_in, open_dir
from whoosh.fields import Schema, TEXT
from whoosh.qparser import QueryParser
from transformers import pipeline
import shutil
from pathlib import Path

# Define a schema for indexing content
schema = Schema(content=TEXT(stored=True))

def clear_huggingface_cache():
    cache_dir = Path.home() / ".cache" / "huggingface" / "transformers"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        print("[INFO] Hugging Face cache cleared.")
    else:
        print("[INFO] No cache found to clear.")

# Step 1: Scrape website content
def scrape_documentation(url):
    print("[STEP] Fetching documentation from:", url)
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        raise Exception(f"[ERROR] Failed to fetch URL: {e}")
    
    soup = BeautifulSoup(response.content, 'html.parser')
    content = '\n'.join([tag.get_text() for tag in soup.find_all(['p', 'h1', 'h2', 'h3'])])
    print("[INFO] Content successfully scraped.")
    print("🔎 Scraped Preview:\n", content[:300], "...\n")
    return content.strip()

# Step 2: Index scraped content
from whoosh import index

def index_content(content):
    print("[STEP] Indexing content...")
    if not os.path.exists("index"):
        os.mkdir("index")
        ix = create_in("index", schema)
    else:
        ix = index.open_dir("index")
    
    writer = ix.writer()
    content_chunks = content.split('\n\n')
    for chunk in content_chunks:
        writer.add_document(content=chunk.strip())
    writer.commit()
    print("[INFO] Content indexed.")

# Step 3: Search indexed content
def search_content(query):
    print(f"[STEP] Searching for: {query}")
    ix = open_dir("index")
    with ix.searcher() as searcher:
        query_parser = QueryParser("content", ix.schema)
        parsed_query = query_parser.parse(query)
        results = searcher.search(parsed_query)

        if results:
            print("[INFO] Relevant context found.")
            return results[0]['content']
        else:
            print("[WARN] No relevant content found.")
            return None

# Step 4: Load QA Model
def load_qa_model():
    print("[STEP] Loading QA model...")
    try:
        model_name = "deepset/minilm-uncased-squad2"
        qa_pipeline = pipeline("question-answering", model=model_name)
        print("[INFO] QA model loaded:", model_name)
        return qa_pipeline
    except Exception as e:
        raise RuntimeError(f"[ERROR] Failed to load model: {e}")

# Step 5: Answer the query
def answer_query(query, context, qa_pipeline):
    try:
        print("[STEP] Running QA model...")
        answer = qa_pipeline(question=query, context=context)
        return answer['answer']
    except Exception as e:
        raise Exception(f"[ERROR] Failed to answer query: {e}")

# Step 6: If no match
def handle_missing_info():
    return "[INFO] Sorry, no relevant information was found in the documentation."

# Main interaction loop
def main():
    print("📘 Welcome to the Documentation Query Agent!")
    
    url = input("🔗 Enter the help documentation URL: ").strip()

    try:
        # Step 1
        content = scrape_documentation(url)

        # Step 2
        index_content(content)

        # Step 3
        qa_pipeline = load_qa_model()

        while True:
            query = input("\n❓ Ask a question (or type 'exit' to quit): ").strip()
            if query.lower() == 'exit':
                print("👋 Exiting. Goodbye!")
                break

            context = search_content(query)

            if context:
                answer = answer_query(query, context, qa_pipeline)
                print(f"✅ Answer: {answer}")
            else:
                print(handle_missing_info())

    except Exception as e:
        print(f"[FATAL ERROR] {e}")

if __name__ == "__main__":
    main()
