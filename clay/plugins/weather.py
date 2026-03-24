# Weather plugin
# requests is used for making HTTP requests
import requests


# Fetches the weather forecast from the National Weather Service API
def get_weather():
    # Try to fetch the weather forecast
    try:
        r = requests.get("https://api.weather.gov/gridpoints/MFL/110,50/forecast")
        data = r.json()
        forecast = data["properties"]["periods"][0]
        return f"{forecast['shortForecast']}, {forecast['temperature']}°F"
    # If there's an error, return a message
    except:
        return "Couldn't fetch weather."
