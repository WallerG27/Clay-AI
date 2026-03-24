# verifyDeps.py is stolen from my PhotoSlop project
## from clay.verifyDeps import ensure_dependencies

# verify + install dependencies first
## ensure_dependencies()

# Calls the appropriate plugin or LLM response based on the user's command
from clay.core.llm_bridge import ask_llm
from clay.core.memory_system import MemorySystem
from clay.router import route_command

# memory system
memory = MemorySystem()

import subprocess
import time

import requests


def ensure_model():
    try:
        r = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "phi", "prompt": "test"},
            timeout=5,
        )
        if r.status_code == 200:
            return
    except:
        pass

    print("Pulling phi model...")
    subprocess.run(["ollama", "pull", "phi"])


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
    # ensure ollama is running
    ensure_ollama_running()
    # ensure model is pulled
    ensure_model()
    # print ready message
    print("Clay: Clay online.")

    # main loop for user input
    while True:
        user_input = input("You: ")

        # route command
        response = route_command(user_input)

        # Only use router if it's NOT a memory-related question
        if response and "remember" not in user_input.lower():
            print(f"Clay: {response}")
            continue

        # exit/quit
        if user_input.lower() in ["exit", "quit"]:
            break

        # store memory
        memory.add(user_input)

        # build context
        context = memory.build_context(user_input)

        # combine everything
        full_prompt = f"""
{context}

User: {user_input}
"""

        # get response
        response = ask_llm(full_prompt)

        # compress memory over time
        memory.compress()

        print(f"Clay: {response}")


# entry point
if __name__ == "__main__":
    main()
