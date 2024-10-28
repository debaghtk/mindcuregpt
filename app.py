import os
import gradio as gr
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from dotenv import load_dotenv
from openai import AsyncOpenAI
import tiktoken
import asyncio
import re
import faiss
import numpy as np
import requests
import tempfile
import random

# Load environment variables from .env file
load_dotenv()

# Set your OpenAI API key from environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")
client = AsyncOpenAI(api_key=openai_api_key)

# Get ElevenLabs API key from environment variable
elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
# Specify the voice ID
voice_id = os.getenv("ELEVENLABS_VOICE_ID")

# Load and process the transcripts
documents = []
folder_path = "transcriptions" if os.getenv("ENVIRONMENT") == "local" else "transcriptions"

def process_transcript_file(file_path):
    with open(file_path, "r", encoding='utf-8') as f:
        content = f.read()

    # Extract video ID from filename
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    video_id = base_name.replace("_transcription", "")
    youtube_link_base = f"https://www.youtube.com/watch?v={video_id}"

    # Parse the transcript content to extract entries with timestamps
    lines = content.splitlines()
    file_documents = []
    for line in lines:
        # Updated regex to match your transcript format
        match = re.match(r"^\[(\d+\.\d+) - (\d+\.\d+)\]\s*(.*)$", line)
        if match:
            start_time = float(match.group(1))
            end_time = float(match.group(2))
            text = match.group(3)
            # Create a Document for each transcript entry
            entry = Document(
                page_content=text,
                metadata={
                    "youtube_link": youtube_link_base,
                    "start_time": start_time,
                    "end_time": end_time,
                    "timestamp_link": f"{youtube_link_base}&t={int(start_time)}",
                    "channel": os.path.basename(os.path.dirname(file_path))  # Add channel name from folder
                }
            )
            file_documents.append(entry)
    return file_documents

# Recursively walk through all subdirectories
for root, dirs, files in os.walk(folder_path):
    for filename in files:
        if filename.endswith(".txt"):
            file_path = os.path.join(root, filename)
            try:
                file_documents = process_transcript_file(file_path)
                documents.extend(file_documents)
                print(f"Processed {file_path}: {len(file_documents)} entries")
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")

# Check documents length
print(f"Total number of documents loaded: {len(documents)}")

# Continue with splitting documents if necessary
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_documents = text_splitter.split_documents(documents)

# Check split_documents length
print(f"Number of documents after splitting: {len(split_documents)}")

# Create OpenAI embeddings and use FAISS as the vector store for retrieval
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

print("Generating embeddings...")

# Generate embeddings and create the vector store
vector_store = FAISS.from_documents(split_documents, embeddings)

# Define the conversation history
conversation_history = []

# Function to count tokens
def count_tokens(messages):
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return sum(len(encoding.encode(message['content'])) for message in messages if 'content' in message)

# Function to truncate messages to fit within the token limit
def truncate_messages(messages, max_tokens):
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    truncated_messages = []
    total_tokens = 0

    # Try to include the system message if it exists
    system_message = next((msg for msg in messages if msg['role'] == 'system'), None)
    if system_message:
        truncated_messages.append(system_message)
        total_tokens = count_tokens([system_message])

    # Always include the latest user message
    user_message = next(msg for msg in reversed(messages) if msg['role'] == 'user')
    truncated_messages.append(user_message)
    total_tokens += count_tokens([user_message])

    # Add other messages if there's room, prioritizing more recent messages
    for message in reversed(messages):
        if message['role'] != 'system' and message != user_message:
            message_tokens = count_tokens([message])
            if total_tokens + message_tokens <= max_tokens:
                truncated_messages.insert(1, message)  # Insert after system message or at the beginning
                total_tokens += message_tokens
            else:
                break

    return truncated_messages

# Function to get dynamic threshold
def get_dynamic_threshold(scores):
    if not scores:
        return 0
    mean_score = sum(scores) / len(scores)
    return mean_score * 0.8  # You can adjust this multiplier

# Function to handle user input and generate response
async def chatbot_response(user_input, history):
    global conversation_history

    max_context_length = 16000  # Leave some room for the response

    system_message = {"role": "system", "content": "You are a compassionate and knowledgeable mental health chatbot. Your purpose is to provide supportive, empathetic responses based on evidence-based mental health practices. You offer general guidance and coping strategies, but you always emphasize that you're not a substitute for professional mental health care. Encourage users to seek help from qualified professionals for specific mental health concerns or in crisis situations."}
    
    messages = [system_message] + conversation_history + [{"role": "user", "content": user_input}]

    relevant_docs = vector_store.similarity_search_with_score(user_input, k=3)
    
    # Calculate dynamic threshold
    scores = [score for _, score in relevant_docs]
    threshold = get_dynamic_threshold(scores)

    filtered_docs = []
    for doc, score in relevant_docs:
        if score > threshold:
            filtered_docs.append((doc, score))
            relevant_text = doc.page_content
            start_time = doc.metadata.get("start_time", 0)
            timestamp_link = doc.metadata.get("timestamp_link", "Link not found")
            messages.append({
                "role": "system",
                "content": f"Consider this relevant information (relevance score: {score:.2f}, timestamp: {start_time}s): {relevant_text}"
            })

            # Check token count and break if we're approaching the limit
            if count_tokens(messages) > max_context_length:
                break

    if filtered_docs:
        messages.append({"role": "system", "content": f"Based on the above information, answer the user's question: {user_input}"})
    else:
        timestamp_link = "No relevant video found."

    # Final truncation to ensure we're within limits
    messages = truncate_messages(messages, max_context_length)

    response = await client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
    answer = response.choices[0].message.content

    if filtered_docs:
        # Get the most relevant document (highest score)
        most_relevant_doc, highest_score = max(filtered_docs, key=lambda x: x[1])
        timestamp_link = most_relevant_doc.metadata.get("timestamp_link", "Link not found")
        start_time = most_relevant_doc.metadata.get("start_time", 0)

        answer_with_link_and_description = (
            f"{answer}\n\n"
            f"Watch the most relevant part of the video here: {timestamp_link}"
        )
    else:
        answer_with_link_and_description = f"{answer}\n\nNo relevant video found for this query."

    # Generate speech using ElevenLabs
    headers = {
        "Accept": "audio/mpeg",
        "xi-api-key": elevenlabs_api_key,
        "Content-Type": "application/json"
    }

    data = {
        "text": answer,
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75
        }
    }

    tts_url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

    response = requests.post(tts_url, json=data, headers=headers)

    if response.status_code == 200:
        # Create a temporary file
        temp_audio_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        temp_audio_file.write(response.content)
        temp_audio_file.close()
        audio_file_path = temp_audio_file.name
    else:
        print(f"Error in ElevenLabs API: {response.status_code}, {response.text}")
        audio_file_path = None

    # Update conversation history
    conversation_history.append({"role": "user", "content": user_input})
    conversation_history.append({"role": "assistant", "content": answer_with_link_and_description})
    conversation_history = truncate_messages(conversation_history, max_context_length // 2)  # Use half the max length for history

    return answer_with_link_and_description, audio_file_path

async def main_chatbot(message, history):
    print(f"Main chatbot function called. Message: {message}")
    response, audio_path = await chatbot_response(message, history)
    history.append({"role": "human", "content": message})
    history.append({"role": "assistant", "content": response})
    return history, audio_path

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    submit = gr.Button("Submit")
    clear = gr.Button("Clear")
    audio_output = gr.Audio(label="Assistant's Voice")

    async def user(user_message, history):
        print(f"User function called. Message: {user_message}")
        return "", history + [[user_message, None]]

    async def bot(history):
        print(f"Bot function called. History length: {len(history)}")
        if not history:
            print("History is empty")
            return history, None
        
        user_message = history[-1][0]
        print(f"User message: {user_message}")
        bot_response, audio_path = await chatbot_response(user_message, history[:-1])
        history[-1][1] = bot_response
        return history, audio_path

    submit.click(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, [chatbot, audio_output]
    )
    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, [chatbot, audio_output]
    )
    clear.click(lambda: None, None, chatbot, queue=False)

demo.launch()

CHUNK_SIZE = 10000  # Adjust as needed

def save_faiss_index(index, filepath):
    faiss.write_index(index, filepath)

def load_faiss_index(filepath):
    return faiss.read_index(filepath)

def process_documents_in_chunks(documents, embeddings, index_filepath):
    index = None
    if os.path.exists(index_filepath):
        index = load_faiss_index(index_filepath)
        print("Loaded existing FAISS index.")
    else:
        # Initialize a new index
        embedding_size = len(embeddings.embed_query("test"))
        index = faiss.IndexFlatL2(embedding_size)
        print("Created new FAISS index.")

    num_chunks = len(documents) // CHUNK_SIZE + int(len(documents) % CHUNK_SIZE > 0)
    for i in range(num_chunks):
        chunk_docs = documents[i*CHUNK_SIZE:(i+1)*CHUNK_SIZE]
        texts = [doc.page_content for doc in chunk_docs]
        metadatas = [doc.metadata for doc in chunk_docs]

        # Generate embeddings for the chunk
        chunk_embeddings = embeddings.embed_documents(texts)

        # Convert embeddings to a numpy array
        embedding_array = np.array(chunk_embeddings).astype("float32")

        # Add embeddings to the index
        index.add(embedding_array)

        # Optionally, store metadata separately (e.g., in a list or a database)
        # For simplicity, we'll assume we can retrieve documents by index

        # Save the index after each chunk
        save_faiss_index(index, index_filepath)
        print(f"Processed chunk {i+1}/{num_chunks} and updated FAISS index.")

def random_response(message, history):
    response = random.choice(["Yes", "No"])
    history.append({"role": "human", "content": message})
    history.append({"role": "assistant", "content": response})
    return history
