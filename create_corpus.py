#!/usr/bin/env python3
#
# Dataset/Corpus Builder for Text SImplification. Using following sources:
# Wikipedia --> Source 
# Klexikon --> 8-14 y.o.
# MiniKlexikon --> just learned how to read

# Hinweis zur Nutzung generativer KI (gemäß Richtlinie der Fakultät für Informatik, Beschluss 015/24 vom 07.2024):
#
# Für diesen Code habe ich den in Visual Studio Code integrierten KI-Assistenten verwendet. 
# Der KI-Einsatz beschränkte sich auf die automatische Vervollständigung von Code-Snippets.
# Die Grundlogik und Strukturierung des Codes enstand in Eigenleistung.
#
# Weiterhin wurden folgende Quellen als Inspiration und Referenz genutzt:
#
# Quelle Python Wikipedia API Wrapper: https://wikipedia-api.readthedocs.io/en/latest/ (l.Z. 06.07.2026)

import argparse
import json
import re
import time
import unicodedata
from pathlib import Path
from urllib.parse import quote, unquote
import requests
import wikipediaapi
from bs4 import BeautifulSoup
from tqdm import tqdm



WIKI_API = "https://de.wikipedia.org/w/api.php"
KLEX_API = "https://klexikon.zum.de/api.php"
MINIKLEX_API = "https://miniklexikon.zum.de/api.php"

USER_AGENT = "TextSimplificationDatasetBuilder (Uni Bachelor Thesis)"

REQUEST_SLEEP_TIME = 0.3 

# HTML elements to avoid
SKIP_CLASSES = [
    "infobox", "navbox", "metadata", "ambox", "toc", "mw-editsection",
    "reference", "reflist", "thumb", "hatnote", "sistersitebox",
    "noprint", "catlinks", "vertical-navbox",
]

session = requests.Session()
session.headers.update({"User-Agent": USER_AGENT})

wiki = wikipediaapi.Wikipedia(
    user_agent=USER_AGENT,
    language="de",
    extra_api_params={"redirects": 1},
)


def api_get(url, params, retries=2):
    params = dict(params)
    params["format"] = "json"

    for i in range(retries):
        try:
            r = session.get(url, params=params, timeout=20)
            r.raise_for_status()
            time.sleep(REQUEST_SLEEP_TIME)
            return r.json()
        except (requests.RequestException, ValueError) as e:
            err = e
            wait = 2.0 * (i + 1)
            print(f"{url} failed ({e}), retry in {wait}seconds")
            time.sleep(wait)


def get_all_titles(api_url, ns=0):
    titles = []
    cont = None

    while True:
        params = {
            "action": "query",
            "list": "allpages",
            "apnamespace": ns,
            "aplimit": "max",
        }
        if cont:
            params["apcontinue"] = cont

        data = api_get(api_url, params)
        for p in data.get("query", {}).get("allpages", []):
            titles.append(p["title"])

        # using only query-continue would not retrieve all pages. For some reason using both keywords solved it. Maybe because of this:
        # https://www.mediawiki.org/wiki/API:Continue
        c = data.get("continue", {})
        query_c = data.get("query-continue", {}).get("allpages", {})

        if "apcontinue" in c:
            cont = c["apcontinue"]
        elif "apcontinue" in query_c:
            cont = query_c["apcontinue"]
        else:
            break

    return titles


def get_rendered_html(api_url, title):
    params = {
        "action": "parse",
        "page": title,
        "prop": "text",
        "redirects": 1,
    }
    data = api_get(api_url, params)
    if "error" in data:
        return None
    return data.get("parse", {}).get("text", {}).get("*")

def find_klexikon_link(mini_html):
    # some MiniKlexikon-Sites link directly to the Klexikon article
    if not mini_html:
        return None

    soup = BeautifulSoup(mini_html, "html.parser")
    for a in soup.find_all("a", href=True):
        href = a["href"].lower()
        # because: "klexikon.zum.de" is a substring of "miniklexikon.zum.de"
        if "klexikon.zum.de" in href and "miniklexikon.zum.de" not in href:
            m = re.search(r"/wiki/([^#?]+)", a["href"])
            if m:
                return unquote(m.group(1)).replace("_", " ")
    return None


def is_may_refer_page(api_url, title):
    # May reference sites e.g.: ("Adler", "Aal", ...) are unwanted
    params = {
        "action": "query",
        "titles": title,
        "prop": "pageprops",
        "redirects": 1,
    }
    data = api_get(api_url, params)
    for page in data.get("query", {}).get("pages", {}).values():
        if "may refer" in page.get("pageprops", {}):
            return True
    return False


def get_wikipedia_lead(candidate_titles):
    # tries candidates until it finds a page that exists and is not a "may refer" page
    for title in candidate_titles:
        page = wiki.page(title)

        try:
            if not page.exists():
                time.sleep(REQUEST_SLEEP_TIME)
                continue
            if is_may_refer_page(WIKI_API, page.title):
                time.sleep(REQUEST_SLEEP_TIME)
                continue
            summary = page.summary
        except wikipediaapi.WikipediaException as e:
            print(f"[warn] wikipedia lookup fuer '{title}' fehlgeschlagen: {e}")
            time.sleep(REQUEST_SLEEP_TIME)
            continue

        time.sleep(REQUEST_SLEEP_TIME)
        return page.title, page.fullurl, clean_text(summary)

    return None, None, None


def strip_unwanted(soup_node):
    for cls in SKIP_CLASSES:
        for tag in soup_node.find_all(class_=cls):
            tag.decompose()
    for tag in soup_node.find_all(["table", "style", "script", "sup"]):
        tag.decompose()
    return soup_node


def get_lead_paragraphs(html):
    #Klexikon: only lead
    if not html:
        return ""

    soup = BeautifulSoup(html, "html.parser")
    content = soup.find("div", class_="mw-parser-output") or soup
    content = strip_unwanted(content)

    paras = []
    for child in content.children:
        tag_name = getattr(child, "name")
        if tag_name in ("h2", "h3", "h4", "h5", "h6"):
            break
        if tag_name == "p":
            txt = child.get_text(" ", strip=True)
            if txt:
                paras.append(txt)

    return clean_text("\n\n".join(paras))


def get_full_text(html):
    # MiniKlexikon: everything 
    if not html:
        return ""

    soup = BeautifulSoup(html, "html.parser")
    content = soup.find("div", class_="mw-parser-output") or soup
    content = strip_unwanted(content)

    paras = [p.get_text(" ", strip=True) for p in content.find_all("p")]
    paras = [p for p in paras if p]

    return clean_text("\n\n".join(paras))


def clean_text(txt):
    txt = unicodedata.normalize("NFC", txt)
    txt = re.sub(r"\[\d+\]", "", txt)
    txt = re.sub(r"[ \t]+", " ", txt)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return txt.strip()


def make_wiki_url(root, title):
    return root + "/wiki/" + quote(title.replace(" ", "_"))


def build_dataset(out_dir="dataset", limit=None, overwrite=False, min_words=5):
    out_dir = Path(out_dir)
    topics_dir = out_dir / "topics"
    topics_dir.mkdir(parents=True, exist_ok=True)

    print("Retrieve MiniKlexikon articles...")
    mini_titles = get_all_titles(MINIKLEX_API)
    print(f"  -> {len(mini_titles)} MiniKlexikon Artikel gefunden")

    print("Retrieve Klexikon articles...")
    klex_titles = get_all_titles(KLEX_API)
    print(f"  -> {len(klex_titles)} Klexikon Artikel gefunden")

    if limit:
        mini_titles = mini_titles[:limit]

    klex_lookup = {t.strip().lower(): t for t in klex_titles}

    manifest = []
    stats = {
        "written": 0,
        "skip_exists": 0,
        "skip_empty": 0,
        "skip_no_klex": 0,
        "skip_no_wiki": 0,
    }

    for mini_title in tqdm(mini_titles, desc="Verarbeite Themen"):
        safe_name = re.sub(r"[^\w\-]", "_", mini_title)[:80]
        if not safe_name:
            safe_name = "untitled"
        out_path = topics_dir / (safe_name + ".json")

        if out_path.exists() and not overwrite:
            stats["skip_exists"] += 1
            continue

        # 1) MiniKlexikon - full text
        mini_html = get_rendered_html(MINIKLEX_API, mini_title)
        mini_full = get_full_text(mini_html)
        if len(mini_full.split()) < min_words:
            stats["skip_empty"] += 1
            continue

        # 2) Klexikon - only lead
        klex_title = find_klexikon_link(mini_html) or klex_lookup.get(mini_title.strip().lower())
        if not klex_title:
            stats["skip_no_klex"] += 1
            continue

        klex_html = get_rendered_html(KLEX_API, klex_title)
        klex_lead = get_lead_paragraphs(klex_html)
        if len(klex_lead.split()) < min_words:
            stats["skip_empty"] += 1
            continue

        klex_entry = {
            "title": klex_title,
            "url": make_wiki_url("https://klexikon.zum.de", klex_title),
            "lead_text": klex_lead,
            "n_tokens": len(klex_lead.split()),
        }

        # 3) Wikipedia - only lead/summary
        candidates = [klex_title]
        if mini_title not in candidates:
            candidates.append(mini_title)

        wiki_title, wiki_url, wiki_lead = get_wikipedia_lead(candidates)
        if not wiki_title:
            stats["skip_no_wiki"] += 1
            continue
        if len(wiki_lead.split()) < min_words:
            stats["skip_empty"] += 1
            continue

        entry = {
            "topic": mini_title,
            "wikipedia": {
                "title": wiki_title,
                "url": wiki_url,
                "lead_text": wiki_lead,
                "n_tokens": len(wiki_lead.split()),
            },
            "klexikon": klex_entry,
            "miniklexikon": {
                "title": mini_title,
                "url": make_wiki_url("https://miniklexikon.zum.de", mini_title),
                "full_text": mini_full,
                "n_tokens": len(mini_full.split()),
            },
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(entry, f, ensure_ascii=False, indent=2)

        manifest.append({"topic": mini_title, "file": str(out_path)})
        stats["written"] += 1

    # keep manifest to avoid loosing old entries when using --overwrite on a subset
    manifest_path = out_dir / "manifest.jsonl"
    mode = "a" if manifest_path.exists() else "w"
    with open(manifest_path, mode, encoding="utf-8") as f:
        for row in manifest:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("\nDone.")
    print(f"  added:          {stats['written']}")
    print(f"  skipped (exists): {stats['skip_exists']}")
    print(f"  skipped (too short):   {stats['skip_empty']}")
    print(f"  skipped (no Klex): {stats['skip_no_klex']}")
    print(f"  skipped (no Wiki): {stats['skip_no_wiki']}")
    print(f"  Output directory: {topics_dir.resolve()}")
    print(f"  Manifest: {manifest_path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="dataset")
    parser.add_argument("--limit", type=int, default=None,
                         help="only retrieve first N MiniKlexikon articles (for testing)")
    parser.add_argument("--overwrite", action="store_true",
                         help="overwrite existing topic files instead of skipping them")
    parser.add_argument("--min-words", type=int, default=30,
                         help="minimum number of words in article")
    args = parser.parse_args()

    build_dataset(
        out_dir=args.out_dir,
        limit=args.limit,
        overwrite=args.overwrite,
        min_words=args.min_words,
    )
