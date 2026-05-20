import os
import sys
from googleapiclient.discovery import build
from dotenv import load_dotenv


def youtube_client():
    """
    Creates a YouTube Data API v3 client.

    The API key is loaded from the .env file to avoid exposing private keys
    in the GitHub repository.
    """

    load_dotenv()

    api_key = os.getenv("YOUTUBE_API_KEY")

    if not api_key:
        sys.stderr.write(
            "Missing YouTube API key. Create a .env file with:\n"
            "YOUTUBE_API_KEY=your_api_key_here\n"
        )
        sys.exit(1)

    try:
        return build("youtube", "v3", developerKey=api_key)

    except Exception as e:
        sys.stderr.write(f"Failed to create YouTube client: {e}\n")
        sys.exit(1)