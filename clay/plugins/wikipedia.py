### import
# re is used for regular expression operations
import re

# requests is used for making HTTP requests
import requests

# Headers for the Wikipedia API requests
HEADERS = {"User-Agent": "ClayAI/1.0"}


# Searches Wikipedia for a given query and returns the first result
def search_wikipedia(query):
    # Removes "what is", "who is", "what are", "who was" from the query
    try:
        query = query.lower()

        for phrase in ["what is", "who is", "what are", "who was"]:
            query = query.replace(phrase, "")

        query = query.strip()
        query = re.sub(r"[^\w\s]", "", query)

        # Looks up the query on Wikipedia and returns the first result
        search_url = "https://en.wikipedia.org/w/api.php"
        search_params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
        }

        # Sends a GET request to the Wikipedia API search endpoint
        search_res = requests.get(search_url, params=search_params, headers=HEADERS)

        # Raises an error if the search request fails
        if search_res.status_code != 200:
            return f"Search failed: {search_res.status_code}"

        # Parses the search results from the JSON response
        search_data = search_res.json()

        # Returns an error if no results are found
        if not search_data.get("query", {}).get("search"):
            return f"No results found for '{query}'."

        # Gets the title of the first result
        title = search_data["query"]["search"][0]["title"]

        # Gets the summary of the first result
        summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title.replace(' ', '_')}"
        summary_res = requests.get(summary_url, headers=HEADERS)

        # Returns an error if the summary request fails
        if summary_res.status_code != 200:
            return f"Summary failed for '{title}'"

        # Parses the summary response from the JSON response
        data = summary_res.json()

        # Returns the summary if available, otherwise returns a message
        if "extract" in data:
            return data["extract"]

        # Returns a message if no summary is available
        return f"No summary available for '{title}'."

        # Returns an error if an exception occurs
    except Exception as e:
        return f"Wikipedia error: {e}"
