import openai
import faiss
import pickle

from vector_database import get_embedding
from dotenv import load_dotenv
import os
import json

# Configure OpenAI API
# Load environment variables from .env file
load_dotenv()


import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Say this is a test",
        }
    ],
    model="gpt-4o-mini",
)

# Load configuration from config.json

CONFIG_FILE = "config.json"

with open(CONFIG_FILE, "r") as config_file:
    config = json.load(config_file)


def build_prompt(query_text, retrieved_results):
    """
    Build a prompt for the LLM using the query and retrieved context.

    Args:
        query_text (str): The user's query.
        retrieved_results (list): Top-N retrieved results with context.

    Returns:
        str: The formatted prompt.
    """
    context = "\n".join([f"{i+1}. {result['metadata']['text']}" for i, result in enumerate(retrieved_results)])
    
    prompt = (
        "Use the provided context to answer the user's query accurately.\n\n"
        f"Context:\n{context}\n\n"
        f"Query: {query_text}\n\n"
        "Answer:"
    )
    return prompt


def generate_response_no_context(prompt, model):
    try:

        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=model,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {e}"


def generate_response(prompt, model):
    """
    Generate a response using the LLM.

    Args:
        prompt (str): The input prompt.
        model (str): The OpenAI model to use.

    Returns:
        str: The generated response.
    """
    try:

        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=model,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {e}"


