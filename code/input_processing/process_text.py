import json
import os
from sentence_transformers import SentenceTransformer

# Load configuration from a JSON file in the main directory
config_path = os.path.join('config.json')

with open(config_path, 'r') as config_file:
    config = json.load(config_file)


# Load SentenceTransformer model for embedding creation
embedding_model = SentenceTransformer(os.environ.get("EMBEDDING_MODEL", config["embedding_model"]))

def process_text_data(text, source=None):
    """
    Process a piece of text: generate embeddings and enrich with metadata.
    
    Args:
        text (str): The text content to process.
        source (str, optional): The source of the text (e.g., "intelligence report", "field notes").
    
    Returns:
        dict: A dictionary with embeddings, original text, and metadata.
    """
    if not text.strip():
        raise ValueError("Text input is empty. Please provide valid text.")
    
    # Generate embeddings for the text
    embeddings = embedding_model.encode(text)
    
    # Metadata enrichment
    metadata = {
        "source": source or "unknown",
        "text_length": len(text),
        "word_count": len(text.split()),
    }
    
    # Combine text, embeddings, and metadata
    processed_data = {
        "text": text,
        "embeddings": embeddings,
        "metadata": metadata,
    }
    
    return processed_data

# Example usage
if __name__ == "__main__":
    # Replace with your text input
    example_text = "Which gospel contains the Sermon on the Mount?"
    example_source = "Bible"

    processed_text_data = process_text_data(example_text, source=example_source)
    
    print("Text:", processed_text_data["text"])
    print("Embeddings:", processed_text_data["embeddings"][:5])  # Print first 5 dimensions
    print("Metadata:", processed_text_data["metadata"])
