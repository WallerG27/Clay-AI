# Memory system for storing and retrieving long-term knowledge

### Imports
# json is used for saving/loading memory data
import json

# os is used for file path operations
import os

# time is used for sleep operations
import time

# faiss is used for vector similarity search
import faiss

# numpy is used for numerical operations
import numpy as np

# requests is used for making HTTP requests
import requests

# SentenceTransformer is used for generating embeddings
from sentence_transformers import SentenceTransformer

# Path to the memory file
MEMORY_FILE = "clay/memory.json"


# Load memory from file, returns an empty dict if the file doesn't exist
def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    return {}


# Save memory to file
def save_memory(data):
    with open(MEMORY_FILE, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------
# Config
# ---------------------------
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "phi"

# ---------------------------
# Embeddings
# ---------------------------
model = SentenceTransformer(
    r"C:clay\models\all-MiniLM-L6-v2\snapshots\c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
)


# Generate an embedding for a given text
def embed(text):
    return model.encode([text])[0].astype("float32")


# ---------------------------
# LLM Call (for summarization)
# ---------------------------
def call_ollama(prompt):
    # Call the local LLM via Ollama API
    try:
        # Send a POST request to the Ollama API with the prompt
        r = requests.post(
            OLLAMA_URL,
            json={"model": MODEL, "prompt": prompt, "stream": False},
            timeout=120,
        )
        # Return the response from the LLM
        return r.json()["response"]
    # Handle any exceptions that occur during the API call
    except Exception as e:
        return f"ERROR: {e}"


# ---------------------------
# Personality System
# ---------------------------
class Personality:
    def __init__(self):
        # Initialize the personality traits with default values
        self.traits = {"skepticism": 0.7, "directness": 0.9, "curiosity": 0.8}

    # Adapt the personality traits based on the intent of the user
    def adapt(self, intent):
        if "question" in intent:
            self.traits["skepticism"] += 0.02

        # Increase curiosity when the user wants to learn
        if "learn" in intent:
            self.traits["curiosity"] += 0.02

        # clamp
        for k in self.traits:
            self.traits[k] = max(0.0, min(1.0, self.traits[k]))

    # Return a prompt that reflects the personality traits
    def to_prompt(self):
        rules = []

        # Add rules for skepticism
        if self.traits["skepticism"] > 0.6:
            rules.append("Challenge assumptions when needed.")

        # Add rules for directness
        if self.traits["directness"] > 0.7:
            rules.append("Be blunt and direct.")

        # Add rules for curiosity
        if self.traits["curiosity"] > 0.7:
            rules.append("Ask deeper questions.")

        # Return the rules as a prompt
        return "\n".join(rules)


# ---------------------------
# Memory System
# ---------------------------
class MemorySystem:
    def __init__(self):
        self.episodic = []
        self.semantic = []
        self.personality = Personality()
        self.index = faiss.IndexFlatL2(384)
        self.structured = load_memory()

    # Add a new piece of text to the episodic memory and semantic index
    def add(self, text):
        vec = embed(text)
        self.episodic.append((text, vec, time.time()))
        self.index.add(np.array([vec]))

        # If the text contains "remember", save it to the structured memory
        if "remember" in text.lower():
            remembered = text.lower().split("remember", 1)[1].strip()

            # clean phrases like "the word"
            remembered = remembered.replace("the word", "").strip()

            self.structured["saved"] = {"value": remembered, "timestamp": time.time()}

            save_memory(self.structured)

    # Retrieve the k nearest neighbors from the episodic memory and semantic index
    def retrieve(self, text, k=3):
        if self.index.ntotal == 0:
            return []

        # Search the index for the k nearest neighbors
        vec = embed(text)
        D, I = self.index.search(np.array([vec]), k)

        # Retrieve the results from the episodic memory and semantic index
        results = []
        for i in I[0]:
            if i < len(self.episodic):
                results.append(self.episodic[i][0])

        # Add the last 2 items from the semantic memory
        results += self.semantic[-2:]
        return results

    # Compress the episodic memory by removing the oldest 5 items and rebuilding the index
    def compress(self):
        if len(self.episodic) < 5:
            return

        # Compress the episodic memory by removing the oldest 5 items and rebuilding the index
        recent = [x[0] for x in self.episodic[-5:]]
        combined = "\n".join(recent)

        # Call the language model to summarize the recent conversation
        prompt = f"""
You are analyzing a conversation.

Extract:
1. Main topic
2. User intent
3. Short summary

Conversation:
{combined}

Respond EXACTLY in this format:

TOPIC: ...
INTENT: ...
SUMMARY: ...
"""

        # Call the language model to summarize the recent conversation
        response = call_ollama(prompt)

        # Parse the summary from the response
        topic, intent, summary = self.parse_summary(response)

        # Add the summary to the semantic memory and adapt the personality
        if summary:
            self.semantic.append(summary)
            self.personality.adapt(intent)

        # remove old episodic
        self.episodic = self.episodic[:-5]
        self.rebuild_index()

    # Parse the summary from the language model's response
    def parse_summary(self, text):
        topic, intent, summary = "", "", ""

        # Extract the topic, intent, and summary from the response
        for line in text.split("\n"):
            if line.startswith("TOPIC:"):
                topic = line.replace("TOPIC:", "").strip()
            elif line.startswith("INTENT:"):
                intent = line.replace("INTENT:", "").strip()
            elif line.startswith("SUMMARY:"):
                summary = line.replace("SUMMARY:", "").strip()

        # Return the parsed topic, intent, and summary
        return topic, intent, summary

    # Rebuild the FAISS index with the current episodic memory
    def rebuild_index(self):
        self.index = faiss.IndexFlatL2(384)
        for text, vec, _ in self.episodic:
            self.index.add(np.array([vec]))

    # Build the context for the language model's response
    def build_context(self, user_input):
        memory = self.retrieve(user_input)
        personality = self.personality.to_prompt()

        # Format the retrieved memory for the context
        formatted_memory = "\n".join([f"- {m}" for m in memory])

        # Include structured context from the semantic memory
        structured_context = ""

        # Pull from structured memory if available
        if "saved" in self.structured:
            structured_context = f"Saved Memory: {self.structured['saved']['value']}"

        # Combine structured context and formatted memory for the final context
        return f"""
        Relevant Memory:
        {structured_context}
        {formatted_memory}

        Behavior rules:
        {personality}
        """
