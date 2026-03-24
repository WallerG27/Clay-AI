# This is a comment
"""
This file verifies the integrity of the requirements.txt file before installing dependencies
I didn't like having to download the requirements.txt file separately
I hated the idea of an injection attack, so I created a security check.
Could I have just asked you to download the requirements.txt separately? yes
"""

### Imports
# hashlib is used to calculate the SHA256 hash of the requirements.txt file
import hashlib

# subprocess is used to run the pip install command to install the dependencies
import subprocess

# sys is used to get the path to the current Python executable, which is needed to run the pip install command
import sys

# pathlib is used to work with file paths in a more convenient way than using strings
from pathlib import Path

# The requirements.txt file variable
DOWNLOAD_FILE = Path("edward/requirements.txt")

# The SHA256 of the requirements.txt
EXPECTED_HASH = "8cc9c69c53f40d19321339ffaf4142a5cf15db40fef01aceb44834854ffe14b4"


# Function to calculate the SHA256 hash of a file
def sha256_file(path):
    # hashlib.sha256() creates a new sha256 hash object
    h = hashlib.sha256()
    # The file is opened in binary mode and read in chunks of 8192 bytes
    # This avoids loading the entire file into memory at once
    # Not needed for this smaller file, but is more efficient for large files long term
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


# Function to ensure the dependencies are installed and the requirements.txt file is valid
def ensure_dependencies():
    # Double check the requirements.txt file is even there
    if not DOWNLOAD_FILE.exists():
        raise RuntimeError("requirements.txt missing")

    # Get the hash of the requirements.txt file
    actual_hash = sha256_file(DOWNLOAD_FILE)

    # Then compares the hash to the expected hash of the file
    # If the hash does not match, then we raise an error and do not install anything
    if actual_hash != EXPECTED_HASH:
        raise RuntimeError("requirements.txt integrity failure")

    # If the file is valid, then we can install the dependencies
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--require-hashes",
            "-r",
            str(DOWNLOAD_FILE),
        ]
    )
