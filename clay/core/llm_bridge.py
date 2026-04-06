### Import
# requests is used for making HTTP requests
import requests

# Model configuration
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.2"

# How many past messages to keep in the active window (each turn = 2 messages: user + assistant)
MAX_HISTORY_MESSAGES = 12  # = 6 exchanges

# System prompt defining Clay's personality
SYSTEM_PROMPT = """You are Clay, a direct and honest AI assistant.

Goals:
- You are meant to challenge the user's thinking and provide feedback to help the user grow.
- You're not harsh, but you're also not soft.
- Use concise, impactful language with an occasional metaphor to help build understanding.

Rules:
- Answer the user's question clearly and concisely.
- Be direct and confident. Skip unnecessary filler phrases.
- If the user's question refers to something said earlier, use it.
- Never repeat the conversation history in your reply.
- You aren't afraid to handle difficult questions or controversial topics with honesty."""


def ask_llm(context: str, history: list[dict], user_input: str) -> str:
    """
    Send a request to Ollama using /api/chat, which supports full message history.

    Args:
        context:    The memory/personality context string from MemorySystem.build_context()
        history:    List of {"role": "user"/"assistant", "content": "..."} dicts
        user_input: The latest user message

    Returns:
        Clay's response as a string.
    """
    # Build the messages list starting with the system prompt
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Inject memory context as a system-level reminder before the conversation
    if context.strip():
        messages.append({"role": "system", "content": f"Relevant Memory:\n{context}"})

    # Trim history to the last MAX_HISTORY_MESSAGES to avoid overwhelming the model
    trimmed_history = history[-MAX_HISTORY_MESSAGES:]
    messages.extend(trimmed_history)

    # Append the current user message
    messages.append({"role": "user", "content": user_input})

    try:
        r = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "messages": messages,
                "stream": False,
            },
            timeout=180,
        )

        data = r.json()

        # /api/chat returns {"message": {"role": "assistant", "content": "..."}}
        if "message" in data and "content" in data["message"]:
            return data["message"]["content"].strip()

        return f"Unexpected response: {data}"

    except Exception as e:
        return f"Local AI error: {e}"
