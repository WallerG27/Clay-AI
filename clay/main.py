# verifyDeps.py is stolen from my PhotoSlop project
##from clay.verifyDeps import ensure_dependencies

# verify + install dependencies first
##ensure_dependencies()

# Calls the appropriate plugin or LLM response based on the user's command
from core.llm_bridge import ask_llm
from core.memory_system import MemorySystem
from router import route_command

# memory system
memory = MemorySystem()

# Conversation history — list of {"role": ..., "content": ...} dicts
# This is what gives Clay turn-by-turn memory within a session
conversation_history = []

# import subprocess for running ollama
# Suppress transformers logging
import logging
import subprocess

# import time for waiting
import time

# import requests for checking ollama status
import requests

logging.getLogger("transformers").setLevel(logging.ERROR)


def ensure_model():
    try:
        r = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama3.2", "prompt": "test"},
            timeout=5,
        )
        if r.status_code == 200:
            return
    except:
        pass

    print("Pulling phi model...")
    subprocess.run(["ollama", "pull", "llama3.2"])


def ensure_ollama_running():
    try:
        # check if already running
        requests.get("http://localhost:11434")
        print("Ollama already running.")
        return
    except:
        print("Starting Ollama...")

    # start ollama serve in background
    subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # wait until it's ready
    for _ in range(10):
        try:
            requests.get("http://localhost:11434")
            print("Ollama started.")
            return
        except:
            time.sleep(1)

    print("Failed to start Ollama.")


# main loop
def main():
    global conversation_history

    # ensure ollama is running
    ensure_ollama_running()
    # ensure model is pulled
    ensure_model()
    # print ready message
    print("Clay: Clay online.")

    # main loop for user input
    while True:
        user_input = input("You: ")

        # exit/quit
        if user_input.lower() in ["exit", "quit"]:
            break

        # Ask the router if any plugin matches
        plugin_result = route_command(user_input)

        # Short-circuit: direct response (e.g. "too short" guard), no LLM needed
        if isinstance(plugin_result, str) and plugin_result.startswith("__DIRECT__:"):
            print(f"Clay: {plugin_result.replace('__DIRECT__:', '').strip()}")
            continue

        # store user message in long-term memory (FAISS + structured)
        memory.add(user_input)

        # build semantic context from memory
        context = memory.build_context(user_input)

        # If the router returned plugin data, prepend it to the user message
        # so Clay has the info but history stays clean
        if plugin_result and "remember" not in user_input.lower():
            augmented_input = f"{plugin_result}\n\nUser asked: {user_input}"
        else:
            augmented_input = user_input

        # get response — pass context, full history, and current input separately
        response = ask_llm(context, conversation_history, augmented_input)

        # update conversation history with both sides of the exchange
        conversation_history.append({"role": "user", "content": user_input})
        conversation_history.append({"role": "assistant", "content": response})

        # compress long-term memory over time
        memory.compress()

        print(f"Clay: {response}")


# entry point
if __name__ == "__main__":
    main()
