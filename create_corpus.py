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
from numpy.random import choice
from jsonl import jsonl
import re
import time
from pathlib import Path
from urllib.parse import quote, unquote
import requests
import wikipediaapi
from bs4 import BeautifulSoup
from tqdm import tqdm
import unicodedata



WIKI_API = "https://de.wikipedia.org/w/api.php"
KLEX_API = "https://klexikon.zum.de/api.php"
MINIKLEX_API = "https://miniklexikon.zum.de/api.php"

#https://www.mediawiki.org/wiki/API:Etiquette user agent according to the api etiquette format
USER_AGENT = "TextSimplificationDatasetBuilder/v1.0 (https://github.com/DerYannis1/Text-Simplification-BA/, yannis.hildebrand@st.ovgu.de) "
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
            request = session.get(url, params=params, timeout=20)
            request.raise_for_status()
            time.sleep(REQUEST_SLEEP_TIME)
            return request.json()
        except (requests.RequestException, ValueError) as e:
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
    # May refer sites e.g.: ("Adler", "Aal", ...) are unwanted
    params = {
        "action": "query",
        "titles": title,
        "prop": "pageprops",
        "redirects": 1,
    }
    data = api_get(api_url, params)
    for page in data.get("query", {}).get("pages", {}).values():
        if "disambiguation" in page.get("pageprops", {}):
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

    return clean_text("\n".join(paras))


def get_full_text(html):
    # MiniKlexikon: everything 
    if not html:
        return ""

    soup = BeautifulSoup(html, "html.parser")
    content = soup.find("div", class_="mw-parser-output") or soup
    content = strip_unwanted(content)

    paras = [p.get_text(" ", strip=True) for p in content.find_all("p")]
    paras = [p for p in paras if p]

    return clean_text("\n".join(paras))


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
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path_train = out_dir /"train.jsonl"
    out_path_test = out_dir / "test.jsonl"
    out_path_validation = out_dir / "validation.jsonl"

    out_path_info = out_dir / "info.json"

    print("Retrieve MiniKlexikon articles...")
    mini_titles = get_all_titles(MINIKLEX_API)
    print(f"  -> {len(mini_titles)} MiniKlexikon Artikel gefunden")

    print("Retrieve Klexikon articles...")
    klex_titles = get_all_titles(KLEX_API)
    print(f"  -> {len(klex_titles)} Klexikon Artikel gefunden")

    if limit:
        mini_titles = mini_titles[:limit]

    klex_lookup = {t.strip().lower(): t for t in klex_titles}

    stats = {
        "written": 0,
        "skip_exists": 0,
        "skip_empty": 0,
        "skip_no_klex": 0,
        "skip_no_wiki": 0,
    }

    existing_topics = []
    if out_path_info.exists():
        with open(out_path_info, "r", encoding="utf-8") as f:
            data = json.load(f)
            existing_topics = [t["topic"] for t in data.get("topic", [])]

    for mini_title in tqdm(mini_titles, desc="Verarbeite Themen"):
        
        if mini_title in existing_topics:
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

        klex_info = {
            "title": klex_title,
            "url": make_wiki_url("https://klexikon.zum.de", klex_title),
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

        info_entry = {
            "topic": mini_title,
            "wikipedia": {
                "title": wiki_title,
                "url": wiki_url,
                "n_tokens": len(wiki_lead.split()),
            },
            "klexikon": klex_info,
            "miniklexikon": {
                "title": mini_title,
                "url": make_wiki_url("https://miniklexikon.zum.de", mini_title),
                "n_tokens": len(mini_full.split()),
            },
        }
        if out_path_info.exists():
            with open(out_path_info, "r+", encoding="utf-8") as f:
                data = json.load(f)
                data["topic"].append(info_entry)
                f.seek(0)
                json.dump(data, f, ensure_ascii=False, indent=2)
        else:
            with open(out_path_info, "w", encoding="utf-8") as f:
                json.dump({"topic": [info_entry]}, f, ensure_ascii=False, indent=2)

        # write train/test/validation files randomized
        # train_odds = 0.7
        # test_odds = 0.15
        # validation_odds = 0.15

        # out_path_data = choice(
        #     [out_path_train, out_path_test, out_path_validation],
        #     p=[train_odds, test_odds, validation_odds]
        # )
        out_path_data = out_dir /"all.jsonl"

        if not out_path_data.exists():
            miniklexikon_json = {"id": mini_title, "source": wiki_lead, "target": mini_full, "level": "MiniKlexikon"}
            jsonl.dump([miniklexikon_json], out_path_data)
            klexikon_json = {"id": mini_title, "source": wiki_lead, "target": klex_lead, "level": "Klexikon"}
            jsonl.append(klexikon_json, out_path_data)
        else:
            miniklexikon_json = {"id": mini_title, "source": wiki_lead, "target": mini_full, "level": "MiniKlexikon"}
            jsonl.append(miniklexikon_json, out_path_data)
            klexikon_json = {"id": mini_title, "source": wiki_lead, "target": klex_lead, "level": "Klexikon"}
            jsonl.append(klexikon_json, out_path_data)

        stats["written"] += 1

    print("\nDone.")
    print(f"  added:          {stats['written']}")
    print(f"  skipped (exists): {stats['skip_exists']}")
    print(f"  skipped (too short):   {stats['skip_empty']}")
    print(f"  skipped (no Klex): {stats['skip_no_klex']}")
    print(f"  skipped (no Wiki): {stats['skip_no_wiki']}")
    print(f"  Output directory: {out_dir.resolve()}")


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
