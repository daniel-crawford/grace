import certifi
import json
import os
from sentence_transformers import SentenceTransformer
import ssl
import tempfile
import whisper
import yt_dlp


ssl._create_default_https_context = ssl.create_default_context
ssl._create_default_https_context().load_verify_locations(certifi.where())

# Load configuration from a JSON file in the main directory
config_path = os.path.join('config.json')

with open(config_path, 'r') as config_file:
    config = json.load(config_file)

# Specify the path to the FFmpeg binary
os.environ["PATH"] += os.pathsep + "/opt/homebrew/bin/ffmpeg"

# Load Whisper model for transcription
whisper_model = whisper.load_model("base.en")

# Load SentenceTransformer model for embedding creation
embedding_model = SentenceTransformer(config["embedding_model"])


def download_audio_from_youtube(youtube_url):
    """
    Download the audio from a YouTube video using yt-dlp.
    
    Args:
        youtube_url (str): URL of the YouTube video.
    
    Returns:
        str: Path to the downloaded audio file.
    """
    temp_dir = tempfile.mkdtemp()
    output_path = os.path.join(temp_dir, "audio.%(ext)s")

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_path,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'ffmpeg_location': "/opt/homebrew/bin/ffmpeg",
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])

    # Return the downloaded audio file path
    audio_file_path = os.path.join(temp_dir, "audio.mp3")
    return audio_file_path


def process_audio(audio_file_path, chunk_size=50):
    """
    Process an audio file: transcribe using Whisper and generate embeddings.
    
    Args:
        audio_file_path (str): Path to the audio file.
    
    Returns:
        dict: A dictionary with transcription, embeddings, and metadata.
    """
    # Transcribe audio file
    result = whisper_model.transcribe(audio_file_path)
    transcription = result['text']

    # Split transcription into chunks
    words = transcription.split()
    transcription_chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    

    # Generate embeddings for each chunk
    chunk_embeddings = [embedding_model.encode(chunk) for chunk in transcription_chunks]
    
    metadata_chunks = []
    for chunk_transcription, chunk_embedding in zip(transcription_chunks, chunk_embeddings):
        metadata_chunks.append(
            {
                "filename": os.path.basename(audio_file_path),
                "filesize": os.path.getsize(audio_file_path),  # in bytes
                "duration": result.get('duration'),  # duration from Whisper (if available)
                "text": chunk_transcription
            }
        )
    
    # Combine transcription, embeddings, and metadata
    processed_data = {
        "transcription": transcription_chunks,
        "embeddings": chunk_embeddings,
        "metadata": metadata_chunks,
    }
    
    return processed_data

# Main function to process YouTube video
def process_youtube_audio(youtube_url, chunk_size=50):
    """
    Download, transcribe, and process audio from a YouTube URL.
    
    Args:
        youtube_url (str): URL of the YouTube video.
    
    Returns:
        dict: Processed data including transcription, embeddings, and metadata.
    """
    print(f"Downloading audio from: {youtube_url}")
    audio_file_path = download_audio_from_youtube(youtube_url)
    
    print("Processing audio...")
    processed_data = process_audio(audio_file_path, chunk_size=chunk_size)
    
    # Cleanup: Remove the downloaded audio file
    os.remove(audio_file_path)
    print("Audio file cleaned up.")
    
    return processed_data

# Example usage
if __name__ == "__main__":
    youtube_url = "https://www.youtube.com/watch?v=wUssm0X_ELw"  # Replace with a valid YouTube URL
    
    processed_data = process_youtube_audio(youtube_url)
    
    print("Transcription:", processed_data["transcription"])
    print("Embeddings:", processed_data["embeddings"][:5])  # Print first 5 dimensions
    print("Metadata:", processed_data["metadata"])
