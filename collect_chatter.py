import os
from datetime import datetime, timedelta, timezone

import httpx
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

REDDIT_TOKEN_URL = "https://www.reddit.com/api/v1/access_token"
REDDIT_URL = "https://oauth.reddit.com/r/dubai/new.json"
KEYWORDS = ("traffic", "accident", "jam", "metro", "szr", "rain", "rta")
DEFAULT_USER_AGENT = "masar-traffic-agent/1.0 (portfolio project)"


def get_supabase_client():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")
    if not url or not key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be configured.")
    return create_client(url, key)


async def fetch_dubai_chatter(limit: int = 25) -> list[dict] | None:
    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    username = os.getenv("REDDIT_USERNAME")
    password = os.getenv("REDDIT_PASSWORD")
    user_agent = os.getenv("REDDIT_USER_AGENT", DEFAULT_USER_AGENT)
    if not all((client_id, client_secret, username, password)):
        print("Reddit collection skipped: configure Reddit OAuth secrets.")
        return None

    params = {"limit": min(max(limit, 1), 100), "raw_json": 1}

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            token_response = await client.post(
                REDDIT_TOKEN_URL,
                data={"grant_type": "password", "username": username, "password": password},
                auth=(client_id, client_secret),
                headers={"User-Agent": user_agent},
            )
            token_response.raise_for_status()
            access_token = token_response.json()["access_token"]

            response = await client.get(
                REDDIT_URL,
                params=params,
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "User-Agent": user_agent,
                },
            )
            response.raise_for_status()
            posts = response.json().get("data", {}).get("children", [])
    except httpx.HTTPStatusError as error:
        print(f"Reddit collection skipped: API returned HTTP {error.response.status_code}.")
        return None
    except (httpx.RequestError, KeyError, ValueError) as error:
        print(f"Reddit collection skipped: {error}.")
        return None

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

    print(f"Reddit collection completed: fetched {len(posts)} posts, found {len(chatter)} matching posts.")
    return chatter


async def collect_and_store_chatter() -> int:
    posts = await fetch_dubai_chatter()
    if posts is None:
        print("Reddit collection finished without storing posts.")
        return 0
    if not posts:
        print("No Reddit posts matched the traffic or weather keywords in the last 24 hours.")
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
