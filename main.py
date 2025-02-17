import faiss
import json
import os 
import pickle
from sentence_transformers import SentenceTransformer
import sys
from tqdm import tqdm
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")


sys.path.append('./code/')
from input_processing.process_audio import process_youtube_audio
from vector_database import add_to_faiss, query_faiss_index, get_embedding, initialize_faiss_index
from rag import build_prompt, generate_response, generate_response_no_context


# Load configuration from a JSON file
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

# File paths for saving the FAISS index and metadata
INDEX_FILE = config.get("index_file", "objects/faiss.index")
METADATA_FILE = config.get("metadata_file", "objects/metadata_store.pkl")
EMBEDDING_DIM = config.get("embedding_dim", 384)
EMBEDDING_MODEL = config.get("embedding_model", "all-MiniLM-L6-v2")
LLM_MODEL = config.get("llm_model", "gpt-4o-mini")

# Build the FAISS index
if os.path.exists(INDEX_FILE) and os.path.exists(METADATA_FILE):
    print("Loading existing FAISS index and metadata...")
    with open(INDEX_FILE, 'rb') as f:
        faiss_index = pickle.load(f)
    with open(METADATA_FILE, 'rb') as f:
        metadata = pickle.load(f)

else:
    print("Building FAISS index from YouTube URLs...")
    texts, embeddings, metadata = [], [], []
    # "Watch" youtube videos and process the audio
    with open('youtube_urls.txt', 'r') as file:
        youtube_urls = file.readlines()
        for url in tqdm(youtube_urls, desc="Processing YouTube URLs"):
            url = url.strip()  # Remove any leading/trailing whitespace
            processed_data = process_youtube_audio(url, chunk_size=50)
            # Do something with processed_data, e.g., save it or print it
            texts += processed_data["transcription"]
            embeddings += processed_data["embeddings"]
            metadata += processed_data["metadata"]
    
    faiss_index = index = faiss.IndexFlatL2(EMBEDDING_DIM) 
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = embedding_model.encode(texts)
    add_to_faiss(faiss_index, metadata, embeddings, metadata)

    with open(INDEX_FILE, 'wb') as f:
        pickle.dump(faiss_index, f)

    with open(METADATA_FILE, 'wb') as f:
        pickle.dump(metadata, f)

def run_query(query_text):
    query_embedding = get_embedding(query_text, "text")

    results = query_faiss_index(faiss_index, metadata, query_embedding, k=10)
    prompt = build_prompt(query_text, results)

    print('='*50)
    print(prompt)
    print('\n\n\n')
    response = generate_response(prompt, LLM_MODEL)
    print(response)
    print('\nCompare to Non-contextual Response:' + '-'*50)
    no_context_response = generate_response_no_context(query_text, LLM_MODEL)
    print(no_context_response)
    print('='*50 + '\n\n\n')




if __name__ == "__main__":
    while True:
        query = input("Enter a query (exit to exit): ")
        if query == "exit":
            break
        run_query(query)
        print('-'*50 + "\n\n\n")
