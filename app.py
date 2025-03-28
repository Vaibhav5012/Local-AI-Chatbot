from flask import Flask, request, jsonify, render_template, send_file
from gpt4all import GPT4All
import ctypes
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor
import speech_recognition as sr
from gtts import gTTS
from pydub import AudioSegment
import io
import asyncio

app = Flask(__name__)
executor = ThreadPoolExecutor(max_workers=8)  # Increased for better concurrency

conversation_history = []  # Store conversation history

@app.route('/')
def home():
    return render_template('index.html')

def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    print("Uncaught exception:", exc_value)
    traceback.print_tb(exc_traceback)

sys.excepthook = handle_exception

def load_model():
    try:
        model = GPT4All(r"C:\Users\bvaib\Desktop\local\DeepSeek-R1-Distill-Llama-8B-Q4_0.gguf")
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

model = load_model()

def generate_response(user_message):
    conversation_history.append(f"User: {user_message}")

    # Trim conversation history to the last 10 interactions
    trimmed_history = conversation_history[-10:]

    prompt = (
        "You are a concise AI assistant. Always respond as 'AI:' and never simulate or pretend to be the user. "
        "Provide direct and relevant answers without extra explanations or unnecessary details."
        + "\n".join(trimmed_history)
        + "\nAI:"
    )

    try:
        response = model.generate(prompt, max_tokens=70)  # Reduced max_tokens for faster response
        response = response.strip().split("User:")[0].strip()  # Ensure it doesn't continue with user lines
        
        # Remove unwanted phrases
        unwanted_phrases = ["let me think", "hmm", "I believe", "analyzing", "thinking", "let's consider"]
        for phrase in unwanted_phrases:
            response = response.replace(phrase, "").strip()
        
        conversation_history.append(f"AI: {response}")
        return response
    except Exception as e:
        return f"Error during model interaction: {str(e)}"

async def async_generate_response(user_message):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, generate_response, user_message)

@app.route('/ask', methods=['POST'])
def ask():
    if model is None:
        return jsonify({'response': 'Model is not loaded properly!'}), 500

    user_message = request.json.get("message", "")

    response = asyncio.run(async_generate_response(user_message))

    return jsonify({"response": response, "history": conversation_history})

@app.route('/speak', methods=['POST'])
def speak():
    text = request.json.get("text", "")

    tts = gTTS(text)
    audio = io.BytesIO()
    tts.write_to_fp(audio)
    audio.seek(0)
    return send_file(audio, mimetype='audio/mpeg')

@app.route('/speech-to-text', methods=['POST'])
def speech_to_text():
    recognizer = sr.Recognizer()
    audio_file = request.files['audio']

    try:
        # Convert to WAV format
        audio = AudioSegment.from_file(audio_file)
        wav_io = io.BytesIO()
        audio.export(wav_io, format='wav')
        wav_io.seek(0)

        # Speech Recognition
        with sr.AudioFile(wav_io) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            return jsonify({'text': text})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

def load_cuda_dll():
    dll_path = r"C:\\Python313\\DLLs\\llamamodel-mainline-cuda-avxonly.dll"
    try:
        ctypes.CDLL(dll_path)
        print("CUDA DLL loaded successfully!")
    except OSError as e:
        print(f"Failed to load DLL: {e}")

load_cuda_dll()

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True, use_reloader=False)
