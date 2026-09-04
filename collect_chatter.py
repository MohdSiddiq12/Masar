import os
from datetime import datetime, timedelta, timezone

import httpx
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

REDDIT_URL = "https://www.reddit.com/r/dubai/new.json"
KEYWORDS = ("traffic", "accident", "jam", "metro", "szr", "rain", "rta")
DEFAULT_USER_AGENT = "masar-traffic-agent/1.0 (portfolio project)"


def get_supabase_client():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be configured.")
    return create_client(url, key)


async def fetch_dubai_chatter(limit: int = 25) -> list[dict]:
    headers = {
        "User-Agent": os.getenv("REDDIT_USER_AGENT", DEFAULT_USER_AGENT),
    }
    params = {"limit": min(max(limit, 1), 100), "raw_json": 1}

    async with httpx.AsyncClient(timeout=20, headers=headers) as client:
        response = await client.get(REDDIT_URL, params=params)
        response.raise_for_status()
        posts = response.json().get("data", {}).get("children", [])

    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
    chatter = []
    for post in posts:
        post_data = post.get("data", {})
        text_content = f"{post_data.get('title', '')} {post_data.get('selftext', '')}".lower()
        created_at = datetime.fromtimestamp(post_data.get("created_utc", 0), timezone.utc)
        if created_at < cutoff or not any(keyword in text_content for keyword in KEYWORDS):
            continue

        chatter.append(
            {
                "reddit_id": post_data["id"],
                "title": post_data.get("title", ""),
                "score": post_data.get("score", 0),
                "post_url": f"https://www.reddit.com{post_data.get('permalink', '')}",
                "snippet": post_data.get("selftext", "")[:500],
                "published_at": created_at.isoformat(),
                "raw_data": post_data,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    return chatter


async def collect_and_store_chatter() -> int:
    posts = await fetch_dubai_chatter()
    if not posts:
        print("No matching Reddit posts from the last 24 hours.")
        return 0

    result = (
        get_supabase_client()
        .table("reddit_chatter")
        .upsert(posts, on_conflict="reddit_id")
        .execute()
    )
    stored_count = len(result.data or posts)
    print(f"Stored {stored_count} Reddit posts in Supabase reddit_chatter.")
    return stored_count


if __name__ == "__main__":
    import asyncio

    asyncio.run(collect_and_store_chatter())
