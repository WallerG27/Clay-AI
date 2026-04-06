# Calls the appropriate plugin or LLM response based on the user's command
# weather is used to get, you'll never guess, the weather for a given location
from plugins.weather import get_weather

# wikipedia is used to search Wikipedia for a given query
from plugins.wikipedia import search_wikipedia


def route_command(text):
    """
    Returns either:
      - A string prefix to inject into the LLM prompt (weather/wikipedia data)
      - None if no plugin matched (main.py will call ask_llm directly)
      - A plain string response for short-circuit cases (e.g. too short)
    """
    text_lower = text.lower()

    # Check for weather-related queries — inject weather data as context
    if "weather" in text_lower:
        return f"[Weather Data]: {get_weather()}"

    # Check for Wikipedia-related queries — inject article summary as context
    if any(x in text_lower for x in ["who is", "what is", "who was", "what are"]):
        return f"[Wikipedia]: {search_wikipedia(text)}"

    # Short query guard
    if len(text.split()) < 1:
        return "__DIRECT__: I have a one word minimum. Say something more interesting."

    # No plugin matched
    return None
