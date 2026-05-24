"""Fetch a list of major US sports venues from Wikidata.

Queries the Wikidata SPARQL endpoint for venues in the US with capacity
>= 5000 and writes a normalized catalog to data/stadiums.json. The
catalog feeds the UI's stadium dropdown; LoRA training is independent.

Usage from project root:

    python scripts/fetch_stadiums.py [--min-capacity 5000] [--limit 500]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import requests


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "data" / "stadiums.json"
WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"
USER_AGENT = "attend-lora/0.1 (https://github.com/your-org/attend-lora; research project)"

SPARQL_TEMPLATE = """
SELECT DISTINCT ?venue ?venueLabel ?cityLabel ?capacity WHERE {{
  ?venue wdt:P31/wdt:P279* wd:Q1076486 .
  ?venue wdt:P17 wd:Q30 .
  ?venue wdt:P1083 ?capacity .
  FILTER(?capacity >= {min_capacity})
  OPTIONAL {{ ?venue wdt:P131 ?city . }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}
ORDER BY DESC(?capacity)
LIMIT {limit}
"""


def slugify(name: str) -> str:
    s = name.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def fetch(min_capacity: int, limit: int) -> list[dict]:
    query = SPARQL_TEMPLATE.format(min_capacity=min_capacity, limit=limit)
    headers = {"Accept": "application/sparql-results+json", "User-Agent": USER_AGENT}
    print(f"Querying Wikidata (min_capacity={min_capacity}, limit={limit})...")
    resp = requests.get(WIKIDATA_ENDPOINT, params={"query": query}, headers=headers, timeout=60)
    resp.raise_for_status()
    bindings = resp.json()["results"]["bindings"]

    seen: set[str] = set()
    catalog: list[dict] = []
    for row in bindings:
        name = row.get("venueLabel", {}).get("value", "").strip()
        if not name or name.startswith("Q"):
            continue
        slug = slugify(name)
        if not slug or slug in seen:
            continue
        seen.add(slug)
        catalog.append({
            "slug": slug,
            "name": name,
            "city": row.get("cityLabel", {}).get("value", "").strip() or None,
            "capacity": int(row["capacity"]["value"]) if "capacity" in row else None,
            "wikidata_id": row["venue"]["value"].rsplit("/", 1)[-1],
        })
    return catalog


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-capacity", type=int, default=5000)
    parser.add_argument("--limit", type=int, default=500)
    args = parser.parse_args()

    try:
        catalog = fetch(args.min_capacity, args.limit)
    except requests.RequestException as exc:
        print(f"ERROR: Wikidata query failed: {exc}")
        return 1

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(catalog, indent=2) + "\n")
    print(f"Wrote {len(catalog)} venues → {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
    if catalog:
        print(f"  largest: {catalog[0]['name']} ({catalog[0]['capacity']:,} seats)")
        print(f"  smallest: {catalog[-1]['name']} ({catalog[-1]['capacity']:,} seats)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
