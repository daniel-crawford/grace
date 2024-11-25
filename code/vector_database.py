import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
import os

# File paths for saving the FAISS index and metadata
INDEX_FILE = "objects/faiss_index.pkl"
METADATA_FILE = "objects/metadata.pkl"


# Load embedding models
text_embedding_model = SentenceTransformer('all-MiniLM-L6-v2')


# Initialize FAISS index
def initialize_faiss_index(dim):
    if os.path.exists(INDEX_FILE):
        print("Loading existing FAISS index...")
        with open(INDEX_FILE, "rb") as f:
            index = pickle.load(f)
    else:
        print("Creating a new FAISS index...")
        index = faiss.IndexFlatL2(dim)  # L2 distance index
    return index

def get_embedding(data, data_type):
    """
    Generate embeddings for text, image, or other data types.

    Args:
        data (str or PIL.Image): The input data.
        data_type (str): The type of the input ('text', 'image').

    Returns:
        np.ndarray: Embedding vector.
    """
    if data_type == "text":
        return text_embedding_model.encode([data])
    elif data_type == "image":
        pass
        #return image_embedding_model.encode(data)
    else:
        raise ValueError(f"Unsupported data type: {data_type}")

def create_faiss_index(dimension):
    """
    Create a new FAISS index with the given dimension.

    Args:
        dimension (int): Dimension of the embeddings.

    Returns:
        faiss.IndexFlatL2: The FAISS index instance.
    """
    index = faiss.IndexFlatL2(dimension)  # L2 similarity
    print(f"Created FAISS index with dimension {dimension}")
    return index

def save_index(index, metadata, index_file, metadata_file):
    """
    Save the FAISS index and metadata to disk.

    Args:
        index (faiss.IndexFlatL2): FAISS index instance.
        metadata (list): List of metadata dictionaries.
        index_file (str): File path to save the FAISS index.
        metadata_file (str): File path to save the metadata.
    """
    faiss.write_index(index, index_file)
    with open(metadata_file, "wb") as f:
        pickle.dump(metadata, f)
    print("FAISS index and metadata saved.")


def add_to_faiss(index, metadata_store, embeddings, metadata_list):
    """
    Add embeddings and metadata to the FAISS index.

    Args:
        index (faiss.IndexFlatL2): FAISS index instance.
        metadata_store (list): List of metadata dictionaries.
        embeddings (np.ndarray): Embedding vectors (N x D).
        metadata_list (list[dict]): Metadata corresponding to embeddings.

    Returns:
        None
    """
    # Add embeddings to FAISS
    index.add(embeddings)

    # Append metadata to the metadata store
    metadata_store.extend(metadata_list)
    print(f"Added {len(embeddings)} entries to the FAISS index.")


def query_faiss_index(index, metadata_store, query_embedding, k=5):
    """
    Query the FAISS index and retrieve the nearest neighbors with metadata.

    Args:
        index (faiss.IndexFlatL2): FAISS index instance.
        metadata_store (list): List of metadata dictionaries.
        query_embedding (np.ndarray): Query vector (1 x D).
        k (int): Number of nearest neighbors to retrieve.

    Returns:
        list: Retrieved entries with metadata and scores.
    """
    # Search the FAISS index
    distances, indices = index.search(query_embedding, k)

    # Collect results
    results = []
    for i, idx in enumerate(indices[0]):
        if idx == -1:  # No result found
            continue
        results.append({"score": distances[0][i], "metadata": metadata_store[idx]})
    return results

if __name__ == "__main__":

    # Step 1: Create or load the FAISS index
    dim = 384  # Dimension of embeddings
    faiss_index = initialize_faiss_index(dim)

    # Step 2: Add data to the index
    texts = [
        "Military strategy involves coordination.",
        "AI tools are used for intelligence.",
        "Logistics planning requires terrain analysis."
    ]
    metadata = [
        {"type": "report_analysis", "source": "file_1.txt", "timestamp": "2024-11-23", "version": "1.0", "text": texts[0]},
        {"type": "report_analysis", "source": "file_2.txt", "timestamp": "2024-11-23", "version": "1.0", "text": texts[1]},
        {"type": "audio_transcription", "source": "audio_1.mp3", "timestamp": "2024-11-23", "version": "1.0", "text": texts[2]}
    ]

    # Encode texts into embeddings
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embedding_model.encode(texts)

    # Add embeddings and metadata to FAISS
    add_to_faiss(faiss_index, metadata, np.array(embeddings), metadata)

    # Save the updated index
    save_index(faiss_index, metadata, INDEX_FILE, METADATA_FILE)

    # Step 3: Query the FAISS index
    query_text = "What is the role of AI in military strategy?"
    query_embedding = embedding_model.encode([query_text])

    results = query_faiss_index(faiss_index, metadata, np.array(query_embedding), k=3)

    # Display results
    print("Query Results:")
    for result in results:
        print(f"Score: {result['score']}")
        print(f"Metadata: {result['metadata']}")
        print()
