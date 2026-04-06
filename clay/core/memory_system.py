# Memory system for storing and retrieving long-term knowledge

### Imports
import json
import os
import time

import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

# Path to the memory file
MEMORY_FILE = "clay/memory.json"


def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    return {}


def save_memory(data):
    with open(MEMORY_FILE, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------
# Config
# ---------------------------
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2"

# Memory tier thresholds
# Episodic items older than this many turns get compressed into semantic summaries
EPISODIC_COMPRESS_AFTER = 5
# Semantic items older than this many compressions get abstracted into topic-only labels
SEMANTIC_ABSTRACT_AFTER = 4

# ---------------------------
# Embeddings
# ---------------------------
model = SentenceTransformer(
    r"C:clay\models\all-MiniLM-L6-v2\snapshots\c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
)


def embed(text):
    return model.encode([text])[0].astype("float32")


# ---------------------------
# LLM Call (for summarization)
# ---------------------------
def call_ollama(prompt):
    try:
        r = requests.post(
            OLLAMA_URL,
            json={"model": MODEL, "prompt": prompt, "stream": False},
            timeout=120,
        )
        return r.json()["response"]
    except Exception as e:
        return f"ERROR: {e}"


# ---------------------------
# Personality System
# ---------------------------
class Personality:
    def __init__(self):
        self.traits = {"skepticism": 0.7, "directness": 0.9, "curiosity": 0.8}

    def adapt(self, intent):
        if "question" in intent:
            self.traits["skepticism"] += 0.02
        if "learn" in intent:
            self.traits["curiosity"] += 0.02
        for k in self.traits:
            self.traits[k] = max(0.0, min(1.0, self.traits[k]))

    def to_prompt(self):
        rules = []
        if self.traits["skepticism"] > 0.6:
            rules.append("Challenge assumptions when needed.")
        if self.traits["directness"] > 0.7:
            rules.append("Be blunt and direct.")
        if self.traits["curiosity"] > 0.7:
            rules.append("Ask deeper questions.")
        return "\n".join(rules)


# ---------------------------
# Memory System
# ---------------------------
class MemorySystem:
    def __init__(self):
        # Tier 1 — Episodic: raw text, full detail, recent turns only
        # Each entry: (text, vector, timestamp)
        self.episodic = []

        # Tier 2 — Semantic: compressed summaries of older episodic chunks
        # Each entry: (summary_text, vector)
        # Example: "User asked about Henry Clay's birthday"
        self.semantic = []

        # Tier 3 — Abstract: topic-only labels distilled from old semantic entries
        # Each entry: plain string topic label
        # Example: "American history / notable people"
        self.abstract = []

        self.personality = Personality()
        self.index = faiss.IndexFlatL2(384)
        self.structured = load_memory()

    # ---------------------------
    # Add
    # ---------------------------
    def add(self, text):
        """Add a new user message to episodic memory."""
        vec = embed(text)
        self.episodic.append((text, vec, time.time()))
        self.index.add(np.array([vec]))

        # Persist explicit "remember X" requests to structured storage
        if "remember" in text.lower():
            remembered = text.lower().split("remember", 1)[1].strip()
            remembered = remembered.replace("the word", "").strip()
            self.structured["saved"] = {"value": remembered, "timestamp": time.time()}
            save_memory(self.structured)

    # ---------------------------
    # Retrieve
    # ---------------------------
    def retrieve(self, text, k=3):
        """
        Return the most relevant memories across all three tiers.
        Recent episodic entries carry full detail; older tiers are progressively vaguer.
        """
        results = []

        # --- Tier 1: Episodic (full detail, recent) ---
        if self.index.ntotal > 0:
            vec = embed(text)
            D, I = self.index.search(np.array([vec]), k)
            for i in I[0]:
                if i < len(self.episodic):
                    results.append(("[Recent] " + self.episodic[i][0]))

        # --- Tier 2: Semantic (summarised, mid-term) ---
        # Search semantic entries by embedding similarity
        if self.semantic:
            sem_vecs = np.array([v for _, v in self.semantic])
            sem_index = faiss.IndexFlatL2(384)
            sem_index.add(sem_vecs)
            query_vec = embed(text)
            k_sem = min(2, len(self.semantic))
            _, sem_I = sem_index.search(np.array([query_vec]), k_sem)
            for i in sem_I[0]:
                if i < len(self.semantic):
                    results.append(("[Summary] " + self.semantic[i][0]))

        # --- Tier 3: Abstract (topic only, oldest) ---
        # Just append the last couple of topic labels — they're vague by design
        for topic in self.abstract[-2:]:
            results.append(("[Topic] " + topic))

        return results

    # ---------------------------
    # Compress
    # ---------------------------
    def compress(self):
        """
        Two-stage compression:
          Stage 1 — Episodic → Semantic: when episodic grows beyond the threshold,
                    summarise the oldest chunk into a natural-language sentence.
          Stage 2 — Semantic → Abstract: when semantic grows beyond its threshold,
                    distil the oldest semantic entries into a bare topic label.
        """
        self._compress_episodic_to_semantic()
        self._compress_semantic_to_abstract()

    def _compress_episodic_to_semantic(self):
        """Summarise the oldest episodic entries into a semantic sentence."""
        if len(self.episodic) < EPISODIC_COMPRESS_AFTER:
            return

        # Take the oldest chunk (everything except the most recent entries)
        chunk = self.episodic[:-EPISODIC_COMPRESS_AFTER]
        if not chunk:
            return

        combined = "\n".join(x[0] for x in chunk)

        prompt = f"""Summarise this conversation fragment in ONE short sentence (10-15 words max).
Focus on the main topic and intent. Be specific — include names and subjects if present.

Conversation:
{combined}

One-sentence summary:"""

        summary = call_ollama(prompt).strip().split("\n")[0]  # take first line only

        if summary and not summary.startswith("ERROR"):
            vec = embed(summary)
            self.semantic.append((summary, vec))

        # Keep only the recent episodic entries and rebuild the index
        self.episodic = self.episodic[-EPISODIC_COMPRESS_AFTER:]
        self._rebuild_episodic_index()

    def _compress_semantic_to_abstract(self):
        """Distil the oldest semantic summaries into a bare topic label."""
        if len(self.semantic) < SEMANTIC_ABSTRACT_AFTER:
            return

        chunk = self.semantic[:-SEMANTIC_ABSTRACT_AFTER]
        if not chunk:
            return

        combined = "\n".join(x[0] for x in chunk)

        prompt = f"""Reduce these summaries to a single short topic label (3-5 words max).
Examples: "American history / politics", "Python programming help", "Personal goals"

Summaries:
{combined}

Topic label:"""

        topic = call_ollama(prompt).strip().split("\n")[0]

        if topic and not topic.startswith("ERROR"):
            self.abstract.append(topic)

        # Drop the compressed semantic entries
        self.semantic = self.semantic[-SEMANTIC_ABSTRACT_AFTER:]

    def _rebuild_episodic_index(self):
        self.index = faiss.IndexFlatL2(384)
        for _, vec, _ in self.episodic:
            self.index.add(np.array([vec]))

    # ---------------------------
    # Build Context
    # ---------------------------
    def build_context(self, user_input):
        """
        Assemble the memory context string to inject into the LLM prompt.
        Tiers are labelled so Clay knows how much to trust each piece.
        """
        memories = self.retrieve(user_input)
        personality = self.personality.to_prompt()

        formatted = "\n".join(f"- {m}" for m in memories)

        structured_context = ""
        if "saved" in self.structured:
            structured_context = f"Pinned: {self.structured['saved']['value']}\n"

        return f"""{structured_context}{formatted}

Behavior rules:
{personality}"""
