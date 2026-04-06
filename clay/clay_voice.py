# -----------------------------
# Imports
# -----------------------------
# logging for debugging
import logging

# subprocess for running ollama
import subprocess

# time for sleep
import time

# keyboard for push-to-talk
import keyboard

# numpy for audio processing
import numpy as np

# pyttsx3 for text-to-speech
import pyttsx3

# requests for HTTP requests
import requests

# scipy for signal processing
import scipy.signal

# sounddevice for audio input
import sounddevice as sd

# whisper for speech-to-text
import whisper

# core modules
from core.llm_bridge import ask_llm
from core.memory_system import MemorySystem
from router import route_command

# -----------------------------
# CONFIG
# -----------------------------
PUSH_TO_TALK_KEY = "space"
SAMPLE_RATE = 44100  # microphone input
CHUNK_DURATION = 0.2  # seconds per small chunk
WHISPER_MODEL = "turbo"  # "tiny", "base", "medium" for accuracy

# -----------------------------
# INIT SYSTEMS
# -----------------------------
memory = MemorySystem()
model = whisper.load_model(WHISPER_MODEL)  # whisper model should be turbo
engine = pyttsx3.init()
logging.getLogger("transformers").setLevel(logging.ERROR)


# -----------------------------
# OLLAMA SETUP
# -----------------------------
def ensure_model():
    # Check if the model is already loaded by trying to generate a response
    try:
        r = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "phi", "prompt": "test"},
            timeout=5,
        )
        # If the response is successful, the model is loaded
        if r.status_code == 200:
            return
    # If the response fails, the model is not loaded, so pull it from the registry
    except:
        pass
    print("Pulling phi model...")
    # Pull the model from the registry
    subprocess.run(["ollama", "pull", "phi"])


# Check if Ollama is running, and start it if not
def ensure_ollama_running():
    try:
        requests.get("http://localhost:11434")
        print("Ollama already running.")
        return
    # If the request fails, Ollama is not running, so start it
    except:
        print("Starting Ollama...")
    # Start Ollama in the background
    subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # Wait for Ollama to start
    for _ in range(10):
        try:
            requests.get("http://localhost:11434")
            print("Ollama started.")
            return
        # If the request fails, Ollama is still starting, so wait and retry
        except:
            time.sleep(1)
    print("Failed to start Ollama.")


# -----------------------------
# AUDIO RECORDING (push-to-talk)
# -----------------------------
def record_audio_while_holding(key=PUSH_TO_TALK_KEY):
    print("\n[Clay listening...]")
    # Record audio chunks while the key is pressed
    audio_chunks = []

    # Record audio in small chunks while the key is pressed
    while keyboard.is_pressed(key):
        chunk = sd.rec(
            int(CHUNK_DURATION * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
        )
        # Wait for the chunk to finish recording before appending
        sd.wait()
        audio_chunks.append(chunk)

    # Concatenate all chunks into a single array
    if not audio_chunks:
        return np.array([])

    # Flatten the array to remove the extra dimension
    audio_array = np.concatenate(audio_chunks, axis=0).flatten()
    print("[Processing...]\n")
    return audio_array


# -----------------------------
# TRANSCRIPTION
# -----------------------------
def transcribe_audio(audio_array):
    # Check if the audio array is empty
    if len(audio_array) == 0:
        return ""

    # Resample to Whisper's target sample rate
    target_sr = whisper.audio.SAMPLE_RATE  # 16kHz
    # Resample if the sample rate doesn't match
    if SAMPLE_RATE != target_sr:
        num_samples = int(len(audio_array) * target_sr / SAMPLE_RATE)
        audio_array = scipy.signal.resample(audio_array, num_samples)

    # Pad or trim the audio array to the model's expected length
    import whisper.audio as w_audio

    # Pad or trim the audio array to the model's expected length
    audio_array = w_audio.pad_or_trim(audio_array)
    # Convert to mel spectrogram and feed into the model
    mel = whisper.log_mel_spectrogram(audio_array).to(model.device)
    # Decode the mel spectrogram to text
    result = model.decode(mel)
    # Return the transcribed text
    return result.text


# -----------------------------
# TTS OUTPUT
# -----------------------------
def speak(text):
    # Convert text to speech
    engine.say(text)
    engine.runAndWait()


# -----------------------------
# CLAY PIPELINE
# -----------------------------
def run_clay(user_input):
    # Always feed everything into memory
    memory.add(user_input)
    # Build context from memory for the LLM
    context = memory.build_context(user_input)

    # Route command first
    response = route_command(user_input)
    # If a response is returned and it's not a "remember" command, return it
    if response and "remember" not in user_input.lower():
        return response

    # If the user input is an exit/quit command, return "exit"
    if user_input.lower() in ["exit", "quit"]:
        return "exit"

    # Otherwise, use LLM
    full_prompt = f"{context}\n\nUser: {user_input}"
    # Pass the full prompt to the LLM and return the response
    response = ask_llm(full_prompt)
    # Compress memory after each response
    memory.compress()
    # Return the response
    return response


# -----------------------------
# MAIN LOOP
# -----------------------------
def main():
    # Ensure Ollama is running and the model is pulled
    ensure_ollama_running()
    ensure_model()
    print(f"Clay: Voice system online. Hold '{PUSH_TO_TALK_KEY.upper()}' to speak.\n")

    # Main loop: wait for the push-to-talk key, record audio, transcribe, and run Clay
    while True:
        keyboard.wait(PUSH_TO_TALK_KEY)
        audio = record_audio_while_holding(PUSH_TO_TALK_KEY)

        # Transcribe the audio to text
        try:
            text = transcribe_audio(audio)
        # If transcription fails, continue to the next iteration
        except Exception as e:
            print("[Error] Transcription failed:", e)
            continue

        # If the transcription is empty, continue to the next iteration
        if not text.strip():
            continue

        # Print the user's text and get Clay's response
        print("You:", text)
        reply = run_clay(text)
        if reply == "exit":
            break

        # Print Clay's response and speak it
        print("Clay:", reply)
        speak(reply)


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    main()
