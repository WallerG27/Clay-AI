# Web server
from flask import Flask

# Sets up the Flask app
app = Flask(__name__)


# App routes, which are the URLs that the app will respond to
@app.route("/")

# Home route, which returns a simple response
def home():
    # Basic message returned by the home route
    return "Edward Assistant Running"


# Main block, which runs the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
