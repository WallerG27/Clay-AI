# Calls the appropriate plugin or LLM response based on the user's command
# llm_bridge is used to call the LLM for long queries
from clay.core.llm_bridge import ask_llm

# weather is used to get, you'll never guess, the weather for a given location
from clay.plugins.weather import get_weather

# wikipedia is used to search Wikipedia for a given query
from clay.plugins.wikipedia import search_wikipedia


# Route the user's command to the appropriate plugin or LLM response
def route_command(text):
    text_lower = text.lower()

    # Check for weather-related queries
    if "weather" in text_lower:
        return get_weather()

    # Check for Wikipedia-related queries
    if any(x in text_lower for x in ["who is", "what is", "who was", "what are"]):
        return search_wikipedia(text)

    # Check for short queries (less than 3 words)
    if len(text.split()) < 3:
        return "Say something more interesting."

    # Default to LLM response for long queries
    return ask_llm(text)
