import os
import time
import pandas as pd
from youtubeclient import youtube_client


SEARCH_TERMS = [
    "Samsung vs iPhone",
    "iPhone vs Samsung",
    "Galaxy S24 vs iPhone 15",
    "Galaxy S25 vs iPhone 16",
    "Android vs iPhone",
    "Samsung Galaxy vs Apple iPhone"
]

MAX_VIDEOS_PER_QUERY = 8
MAX_COMMENT_PAGES_PER_VIDEO = 10


def search_videos(client, query, max_results=8):
    """
    Searches YouTube for videos matching the query.
    """

    response = client.search().list(
        q=query,
        part="snippet",
        type="video",
        order="viewCount",
        maxResults=max_results
    ).execute()

    videos = []

    for item in response.get("items", []):
        videos.append({
            "search_query": query,
            "video_id": item["id"]["videoId"],
            "video_title": item["snippet"]["title"],
            "channel_title": item["snippet"]["channelTitle"],
            "published_at": item["snippet"]["publishedAt"]
        })

    return videos


def get_video_statistics(client, video_ids):
    """
    Collects statistics for videos in batches.
    YouTube allows up to 50 video IDs in one videos.list request.
    """

    all_stats = []

    for i in range(0, len(video_ids), 50):
        batch_ids = video_ids[i:i + 50]

        response = client.videos().list(
            part="snippet,statistics",
            id=",".join(batch_ids)
        ).execute()

        for item in response.get("items", []):
            snippet = item.get("snippet", {})
            stats = item.get("statistics", {})

            all_stats.append({
                "video_id": item["id"],
                "video_title": snippet.get("title"),
                "channel_title": snippet.get("channelTitle"),
                "published_at": snippet.get("publishedAt"),
                "view_count": int(stats.get("viewCount", 0)),
                "like_count": int(stats.get("likeCount", 0)),
                "comment_count": int(stats.get("commentCount", 0))
            })

    return pd.DataFrame(all_stats)


def get_comments_for_video(client, video_id, max_pages=10):
    """
    Collects top-level comments and visible replies from a YouTube video.
    The reply relationships will be used for network analysis.
    """

    comments = []
    next_page_token = None
    page_count = 0

    while page_count < max_pages:
        try:
            response = client.commentThreads().list(
                videoId=video_id,
                part="snippet,replies",
                maxResults=100,
                pageToken=next_page_token,
                textFormat="plainText"
            ).execute()

            for item in response.get("items", []):
                top_comment = item["snippet"]["topLevelComment"]
                top_snippet = top_comment["snippet"]

                top_comment_id = top_comment["id"]
                top_author = top_snippet.get("authorDisplayName", "Unknown")

                comments.append({
                    "video_id": video_id,
                    "comment_id": top_comment_id,
                    "parent_id": None,
                    "author_name": top_author,
                    "comment_text": top_snippet.get("textDisplay", ""),
                    "like_count": top_snippet.get("likeCount", 0),
                    "published_at": top_snippet.get("publishedAt"),
                    "is_reply": False,
                    "reply_to_author": None
                })

                if "replies" in item:
                    for reply in item["replies"]["comments"]:
                        reply_snippet = reply["snippet"]

                        comments.append({
                            "video_id": video_id,
                            "comment_id": reply["id"],
                            "parent_id": top_comment_id,
                            "author_name": reply_snippet.get("authorDisplayName", "Unknown"),
                            "comment_text": reply_snippet.get("textDisplay", ""),
                            "like_count": reply_snippet.get("likeCount", 0),
                            "published_at": reply_snippet.get("publishedAt"),
                            "is_reply": True,
                            "reply_to_author": top_author
                        })

            next_page_token = response.get("nextPageToken")

            if not next_page_token:
                break

            page_count += 1
            time.sleep(0.2)

        except Exception as e:
            print(f"Could not collect comments for video {video_id}: {e}")
            break

    return comments


def main():
    os.makedirs("data/raw", exist_ok=True)

    client = youtube_client()

    all_videos = []

    for term in SEARCH_TERMS:
        print(f"Searching videos for: {term}")
        videos = search_videos(client, term, MAX_VIDEOS_PER_QUERY)
        all_videos.extend(videos)

    videos_df = pd.DataFrame(all_videos)
    videos_df = videos_df.drop_duplicates(subset=["video_id"])

    video_stats_df = get_video_statistics(client, videos_df["video_id"].tolist())
    video_stats_df.to_csv("data/raw/youtube_videos.csv", index=False)

    print(f"Saved {len(video_stats_df)} videos to data/raw/youtube_videos.csv")

    all_comments = []

    for video_id in video_stats_df["video_id"]:
        print(f"Collecting comments for video: {video_id}")
        comments = get_comments_for_video(client, video_id, MAX_COMMENT_PAGES_PER_VIDEO)
        all_comments.extend(comments)

    comments_df = pd.DataFrame(all_comments)
    comments_df.to_csv("data/raw/youtube_comments.csv", index=False)

    print(f"Saved {len(comments_df)} comments to data/raw/youtube_comments.csv")


if __name__ == "__main__":
    main()