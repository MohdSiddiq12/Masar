import os
from datetime import datetime, timedelta, timezone

import httpx
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
REDDIT_URL = "https://www.reddit.com/r/dubai/new.json"
USER_AGENT = os.getenv(
    "REDDIT_USER_AGENT",
    "python:masar.portfolio.dev:v1.0 (by /u/your_reddit_username)",
)
KEYWORDS = {
    "traffic",
    "accident",
    "jam",
    "metro",
    "szr",
    "rain",
    "rta",
    "congestion",
    "road",
    "sheikh zayed",
}


def fetch_dubai_chatter(limit: int = 30) -> list[dict] | None:
    headers = {"User-Agent": USER_AGENT}
    params = {"limit": min(max(limit, 1), 100), "raw_json": 1}

    try:
        import time
        from masar.api_report import record_call
        started = time.perf_counter()
        with httpx.Client(timeout=15, headers=headers) as client:
            response = client.get(REDDIT_URL, params=params)
            response.raise_for_status()
            posts = response.json().get("data", {}).get("children", [])
        record_call("reddit", "new_posts", {"url": REDDIT_URL, "params": params}, {"post_count": len(posts)}, duration_ms=(time.perf_counter() - started) * 1000)
    except httpx.HTTPStatusError as error:
        from masar.api_report import record_call
        record_call("reddit", "new_posts", {"url": REDDIT_URL, "params": params}, status="error", error=error)
        print(f"Reddit request failed with HTTP {error.response.status_code}.")
        return None
    except (httpx.RequestError, ValueError) as error:
        print(f"Reddit request failed: {error}.")
        return None

    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
    matched = []
    fetched_at = datetime.now(timezone.utc).isoformat()

    for post in posts:
        data = post.get("data", {})
        created = datetime.fromtimestamp(data.get("created_utc", 0), tz=timezone.utc)
        text = f"{data.get('title', '')} {data.get('selftext', '')}".lower()
        if created < cutoff or not any(keyword in text for keyword in KEYWORDS):
            continue

        matched.append(
            {
                "reddit_id": data["id"],
                "title": data.get("title", ""),
                "score": data.get("score", 0),
                "post_url": f"https://reddit.com{data.get('permalink', '')}",
                "snippet": (data.get("selftext") or "")[:300],
                "published_at": created.isoformat(),
                "fetched_at": fetched_at,
                "raw_data": data,
            }
        )

    print(f"Reddit collection completed: fetched {len(posts)} posts, found {len(matched)} matching posts.")
    return matched


def main() -> int:
    print("Starting Reddit collection (public JSON endpoint)...")

    if not SUPABASE_URL or not SUPABASE_KEY:
        print("Missing Supabase credentials; no posts were stored.")
        return 0

    posts = fetch_dubai_chatter()
    if posts is None:
        print("Reddit collection finished without storing posts.")
        return 0
    if not posts:
        print("No matching posts found in the last 24 hours.")
        return 0

    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    for post in posts:
        from masar.api_report import measured_call
        measured_call("supabase", "reddit_chatter.upsert", {"table": "reddit_chatter", "on_conflict": "reddit_id", "post": post}, lambda: supabase.table("reddit_chatter").upsert(post, on_conflict="reddit_id").execute())

    print(f"Stored {len(posts)} posts in reddit_chatter.")
    return len(posts)


if __name__ == "__main__":
    main()
