"""
Step 1 - Turn a spoken sentence into a structured search query.

    "I'm looking for running shoes under Rs 5000"
        -> {"category": "running_shoes", "budget": 5000}

Default path : a deterministic regex/keyword parser -- no credentials, fully
               offline, and what every reported result was produced with.
Optional path: one Claude API call, used only when ANTHROPIC_API_KEY is set.
"""

import os
import re
import json

from .config import CATEGORIES

# Words a user might say -> our canonical category id.
_SYNONYMS = {
    "running_shoes": ["running shoe", "running shoes", "sneaker", "trainer", "shoe"],
    "headphones": ["headphone", "earphone", "earbud", "headset"],
    "coffee": ["coffee", "espresso", "arabica", "beans"],
    "laptops": ["laptop", "notebook", "macbook", "ultrabook"],
    "backpacks": ["backpack", "rucksack", "bag"],
    "smartwatches": ["smartwatch", "smart watch", "fitness band", "watch"],
    "yoga_mats": ["yoga mat", "exercise mat", "fitness mat"],
    "sunglasses": ["sunglass", "shades", "goggles"],
    "cookware": ["cookware", "pan", "cooker", "skillet", "pot"],
    "books": ["book", "novel", "paperback"],
}

_SYSTEM = (
    "You extract shopping intent from a short spoken request. "
    "Return ONLY a JSON object with keys 'category' and 'budget'. "
    f"'category' must be one of: {CATEGORIES}. "
    "'budget' is the max price in INR as an integer, or null if not stated."
)


def _parse_with_regex(text: str) -> dict:
    low = text.lower()

    category = None
    for cat, words in _SYNONYMS.items():
        if any(w in low for w in words):
            category = cat
            break

    budget = None
    # matches: 5000 / rs 5000 / ₹5,000 / 5k / under 4200
    m = re.search(r"(?:rs\.?|₹|inr)?\s*([\d,]+)\s*(k?)", low)
    if m:
        num = int(m.group(1).replace(",", ""))
        if m.group(2) == "k":
            num *= 1000
        budget = num

    return {"category": category, "budget": budget}


def _parse_with_claude(text: str) -> dict:
    from anthropic import Anthropic

    client = Anthropic()  # reads ANTHROPIC_API_KEY from env
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=100,
        system=_SYSTEM,
        messages=[{"role": "user", "content": text}],
    )
    raw = msg.content[0].text.strip()
    raw = re.sub(r"^```(json)?|```$", "", raw, flags=re.MULTILINE).strip()
    data = json.loads(raw)
    cat = data.get("category")
    if cat not in CATEGORIES:
        cat = None
    return {"category": cat, "budget": data.get("budget")}


def parse_query(text: str, use_llm: bool | None = None) -> dict:
    """Return {'category': str|None, 'budget': int|None} for a sentence."""
    if use_llm is None:
        use_llm = bool(os.environ.get("ANTHROPIC_API_KEY"))

    if use_llm:
        try:
            return _parse_with_claude(text)
        except Exception as e:  # network / key / parse issue -> graceful fallback
            print(f"[query_parser] Claude call failed ({e}); using regex fallback.")

    return _parse_with_regex(text)


if __name__ == "__main__":
    for q in [
        "I'm looking for running shoes under Rs 5000",
        "show me some noise cancelling headphones below 3k",
        "need a gaming laptop",
    ]:
        print(f"{q!r:55} -> {parse_query(q, use_llm=False)}")
