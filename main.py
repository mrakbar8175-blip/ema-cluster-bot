import requests
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- CONFIGURATION ---
# Use environment variables for security (fallback to placeholder if not set)
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "PASTE_YOUR_DISCORD_WEBHOOK_URL_HERE")

# Compile regex patterns with word boundaries (\b) to prevent substring false positives
# e.g., \bai\b will match "AI" but NOT "mail" or "paid"
KEYWORDS = ["ai", "gpt", "saas", "tool", "show hn", "startup", "agent", "automation"]
COMPILED_KEYWORDS = [re.compile(r'\b' + re.escape(kw) + r'\b', re.IGNORECASE) for kw in KEYWORDS]

def fetch_item(story_id):
    """Helper function to fetch a single HN item."""
    try:
        item_url = f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json"
        response = requests.get(item_url, timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"⚠️ Error fetching item {story_id}: {e}")
        return None

def get_top_tech_stories():
    print("🕵️‍♂️ Scanning Hacker News for new AI & SaaS tools...")
    url = "https://hacker-news.firebaseio.com/v0/topstories.json"

    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        story_ids = response.json()[:30] # Check top 30
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to fetch top stories: {e}")
        return []

    found_tools = []

    # Use ThreadPoolExecutor to fetch all 30 items concurrently (Massive speed boost!)
    with ThreadPoolExecutor(max_workers=15) as executor:
        future_to_id = {executor.submit(fetch_item, sid): sid for sid in story_ids}

        for future in as_completed(future_to_id):
            item = future.result()
            if not item:
                continue

            title = item.get('title')
            if not title:
                continue # Skip items without titles (like comments or jobs)

            # Check if the title matches our keywords using Regex word boundaries
            if any(pattern.search(title) for pattern in COMPILED_KEYWORDS):
                found_tools.append({
                    'title': title,
                    'url': item.get('url', f"https://news.ycombinator.com/item?id={item.get('id')}"),
                    'score': item.get('score', 0)
                })

    # Sort by score descending so the best tools appear first
    found_tools.sort(key=lambda x: x['score'], reverse=True)
    return found_tools

def send_to_discord(tools):
    if not tools:
        print("No new tools found today.")
        return

    # Format the message beautifully for Discord using Markdown
    description = "*Found on Hacker News. Get them before everyone else!*\n\n"

    for tool in tools[:5]: # Send the top 5 tools
        description += f"🔹 **[{tool['title']}]({tool['url']})**\n"
        description += f"⭐ Community Score: {tool['score']}\n\n"

    description += "💡 *Want early access to more tools? DM the admin.*"

    # Discord uses "Embeds" to make messages look like beautiful cards
    payload = {
        "username": "AI Tool Hunter",
        "embeds": [{
            "title": "🚀 TOP NEW AI & SAAS TOOLS TODAY 🚀",
            "description": description,
            "color": 5814783 # A nice tech-blue color
        }]
    }

    try:
        response = requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=10)
        response.raise_for_status()
        print("✅ Alerts sent to Discord!")
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to send to Discord: {e}")

if __name__ == "__main__":
    tools = get_top_tech_stories()
    send_to_discord(tools)