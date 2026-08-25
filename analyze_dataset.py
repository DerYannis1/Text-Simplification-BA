#!/usr/bin/env python3
"""
Umfassende Analyse des Rohkorpus aus build_corpus.py (Standard-Schema:
{"id": ..., "source": ..., "target": ..., "level": "Klexikon"|"MiniKlexikon"}).

Fasst alle Diagnosen zusammen, die sich im Projektverlauf als relevant
herausgestellt haben:
  - Grundstruktur (Anzahl, Niveau-Verteilung, Vollstaendigkeit der Paare)
  - Laengen-/Kompressions-Statistiken (Zeichen, Woerter, Saetze)
  - Grounding-Score (Wortueberlappung Quelle<->Ziel, siehe
    prepare_training_data.py) -- Kern-Diagnose fuer Halluzinationsrisiko
  - Duplikate & Near-Duplikate (exakt, Gross-/Kleinschreibung)
  - Boilerplate-Erkennung (MiniKlexikon-Navigationsfloskel)
  - Wiki-Markup-Reste in der Quelle
  - Lesbarkeit (WSTF, LIX, Flesch-Amstad) fuer Quelle und Ziel, pro Niveau
  - Wortschwierigkeit (AoA-Perzentile), falls GPT-AoA-Ressource vorhanden
  - Split-Vollstaendigkeit (train/val/test), falls diese Dateien im selben
    Ordner liegen

Ausgabe: analysis_report.md, per_record_metrics.csv, sowie 4 PNG-Diagramme
im --output-dir.

Beispielaufruf:
    python analyze_dataset.py --input dataset_mt5/all.jsonl
"""

import argparse
import csv
import json
import logging
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

try:
    import textstat
    from textstat.backend import counts as _ts_counts
    _HAS_TEXTSTAT = True
except ImportError:
    _HAS_TEXTSTAT = False
    logger.warning("textstat nicht installiert -- Lesbarkeitsmetriken werden uebersprungen "
                    "(`pip install textstat`).")

WORD_RE = re.compile(r"[A-Za-zÄÖÜäöüß]{4,}")
BOILERPLATE_RE = re.compile(
    r"Mehr über\s*[„\"]?[^\n„\"]*[“\"]?\s*findest du auch im "
    r"Kinder-Lexikon Klexikon und bei der Such-Maschine Frag Finn",
    re.IGNORECASE,
)
MARKUP_PATTERNS = {
    "ref_brackets [1]": re.compile(r"\[\d+\]"),
    "doppelte_klammern {{": re.compile(r"\{\{"),
    "wiki_links [[": re.compile(r"\[\["),
    "html_entities": re.compile(r"&[a-z]+;"),
    "section_headers ==": re.compile(r"==[^=]+=="),
    "html_tags": re.compile(r"<[a-zA-Z/][^>]*>"),
}


# ---------------------------------------------------------------------------
# Daten laden
# ---------------------------------------------------------------------------

def load_jsonl(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def get_source_target(record):
    """Unterstuetzt sowohl das Rohkorpus-Schema (source/target) als auch das
    T5-formatierte Schema aus prepare_training_data.py (input_text/target_text)."""
    if "source" in record and "target" in record:
        return record["source"], record["target"]
    source = re.sub(r"^vereinfache\s+<[^>]+>:\s*", "", record.get("input_text", ""), count=1)
    return source, record.get("target_text", "")


# ---------------------------------------------------------------------------
# Metriken pro Datensatz
# ---------------------------------------------------------------------------

def grounding_score(source, target):
    """Anteil der Inhaltswoerter (>=4 Zeichen) im Zieltext, die auch in der
    Quelle vorkommen -- dieselbe Definition wie in prepare_training_data.py."""
    source_words = set(w.lower() for w in WORD_RE.findall(source))
    target_words = set(w.lower() for w in WORD_RE.findall(target))
    if not target_words:
        return 0.0
    return len(target_words & source_words) / len(target_words)


def split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


def flesch_amstad(text):
    """Flesch-AMSTAD (Amstad, 1978), die deutsche Adaption der Formel
    (FRE_de = 180 - ASL - 58.5*ASW), direkt mit lang='de' implementiert statt
    ueber den globalen, veraenderlichen textstat.set_lang()-Zustand -- siehe
    evaluate_mt5.py fuer denselben Ansatz und die Verifikation gegen
    textstat.flesch_reading_ease()."""
    if not _HAS_TEXTSTAT or not text.strip():
        return None
    try:
        n_sentences = _ts_counts.count_sentences(text)
        n_words = _ts_counts.count_words(text)
        n_syllables = _ts_counts.count_syllables(text, "de")
        if n_sentences == 0 or n_words == 0:
            return None
        asl = n_words / n_sentences
        asw = n_syllables / n_words
        return 180 - asl - 58.5 * asw
    except Exception:
        return None


def safe_readability(text):
    if not _HAS_TEXTSTAT or not text.strip():
        return {"wstf": None, "lix": None, "flesch_amstad": None}
    try:
        return {
            "wstf": textstat.wiener_sachtextformel(text, variant=1),
            "lix": textstat.lix(text),
            "flesch_amstad": flesch_amstad(text),
        }
    except Exception:
        return {"wstf": None, "lix": None, "flesch_amstad": None}


def load_aoa_lookup(path):
    if not path or not Path(path).exists():
        return None
    try:
        import pandas as pd
        suffix = Path(path).suffix.lower()
        df = pd.read_excel(path) if suffix in (".xlsx", ".xls") else pd.read_csv(path, encoding="utf-8-sig")
        word_col = "Word" if "Word" in df.columns else df.columns[0]
        value_col = "GPT_AoA_ft" if "GPT_AoA_ft" in df.columns else "AoAestimate"
        df = df[[word_col, value_col]].dropna()
        df["_key"] = df[word_col].astype(str).str.strip().str.lower()
        return df.groupby("_key")[value_col].min().astype(float).to_dict()
    except Exception as e:
        logger.warning("AoA-Ressource konnte nicht geladen werden (%s) -- wird uebersprungen.", e)
        return None


def compute_record_metrics(record, aoa_lookup):
    source, target = get_source_target(record)
    src_words = source.split()
    tgt_words = target.split()

    entry = {
        "id": record.get("id"),
        "level": record.get("level"),
        "source_chars": len(source),
        "target_chars": len(target),
        "source_words": len(src_words),
        "target_words": len(tgt_words),
        "source_sentences": len(split_sentences(source)),
        "target_sentences": len(split_sentences(target)),
        "compression_ratio": (len(tgt_words) / len(src_words)) if src_words else None,
        "grounding_score": grounding_score(source, target),
        "has_boilerplate": bool(BOILERPLATE_RE.search(target)),
        "has_markup_artifact": any(p.search(source) for p in MARKUP_PATTERNS.values()),
    }

    src_read = safe_readability(source)
    tgt_read = safe_readability(target)
    entry.update({f"source_{k}": v for k, v in src_read.items()})
    entry.update({f"target_{k}": v for k, v in tgt_read.items()})

    if aoa_lookup is not None:
        tgt_content_words = WORD_RE.findall(target)
        aoas = [aoa_lookup[w.lower()] for w in tgt_content_words if w.lower() in aoa_lookup]
        entry["target_mean_word_aoa"] = statistics.mean(aoas) if aoas else None
        entry["target_p90_word_aoa"] = sorted(aoas)[int(len(aoas) * 0.9)] if len(aoas) >= 10 else None

    return entry


# ---------------------------------------------------------------------------
# Aggregations-Hilfsfunktionen
# ---------------------------------------------------------------------------

def describe(values, label, unit=""):
    values = [v for v in values if v is not None]
    if not values:
        return f"{label}: keine Werte verfuegbar"
    s = sorted(values)
    n = len(s)
    return (
        f"{label}: n={n}  min={s[0]:.2f}{unit}  p10={s[n//10]:.2f}{unit}  "
        f"median={statistics.median(s):.2f}{unit}  mean={statistics.mean(s):.2f}{unit}  "
        f"p90={s[int(n*0.9)]:.2f}{unit}  max={s[-1]:.2f}{unit}"
    )


def find_duplicates(records):
    ids = [r.get("id", "") for r in records]
    id_counts = Counter(ids)
    exact_dup = {k: v for k, v in id_counts.items() if v > 2}  # >2, da 1 Klexikon+1 MiniKlexikon normal ist

    norm_groups = defaultdict(list)
    for id_ in set(ids):
        norm = str(id_).lower().replace("ß", "ss").replace("-", " ").strip()
        norm_groups[norm].append(id_)
    near_dup = {k: v for k, v in norm_groups.items() if len(v) > 1}

    return exact_dup, near_dup


def check_level_pairing(records):
    by_id = defaultdict(set)
    for r in records:
        by_id[r.get("id")].add(r.get("level"))

    complete_pairs = sum(1 for levels in by_id.values() if {"Klexikon", "MiniKlexikon"} <= levels)
    only_klexikon = sum(1 for levels in by_id.values() if levels == {"Klexikon"})
    only_mini = sum(1 for levels in by_id.values() if levels == {"MiniKlexikon"})
    return {
        "unique_ids": len(by_id),
        "complete_pairs": complete_pairs,
        "only_klexikon": only_klexikon,
        "only_mini": only_mini,
    }


def check_split_files(input_path):
    """Prueft, ob train/val/test.jsonl im selben Ordner liegen, und ob sich
    Artikel-IDs zwischen den Splits ueberschneiden (Data Leakage)."""
    folder = Path(input_path).parent
    split_files = {name: folder / f"{name}.jsonl" for name in ("train", "val", "test")}
    existing = {name: p for name, p in split_files.items() if p.exists()}
    if not existing:
        return None

    ids_by_split = {}
    for name, p in existing.items():
        recs = load_jsonl(p)
        ids_by_split[name] = set(r.get("id") for r in recs)

    overlaps = {}
    names = list(ids_by_split.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = ids_by_split[names[i]] & ids_by_split[names[j]]
            if overlap:
                overlaps[f"{names[i]}<->{names[j]}"] = len(overlap)

    return {"splits_found": list(existing.keys()), "sizes": {k: len(v) for k, v in ids_by_split.items()},
            "overlaps": overlaps}


# ---------------------------------------------------------------------------
# Diagramme
# ---------------------------------------------------------------------------

def plot_histogram(values, title, xlabel, path, bins=40, by_level=None):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if by_level:
        for level, vals in by_level.items():
            vals = [v for v in vals if v is not None]
            if vals:
                ax.hist(vals, bins=bins, alpha=0.6, label=level)
        ax.legend()
    else:
        vals = [v for v in values if v is not None]
        ax.hist(vals, bins=bins, color="#4C72B0")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Anzahl")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(output_dir, records, metrics, pairing, exact_dup, near_dup, split_info, args):
    by_level = defaultdict(list)
    for m in metrics:
        by_level[m["level"]].append(m)

    lines = ["# Datensatz-Analysebericht", "", f"Quelle: `{args.input}`  |  n={len(records)} Zeilen", ""]

    # --- Grundstruktur ---
    lines.append("## Grundstruktur")
    level_counts = Counter(m["level"] for m in metrics)
    for level, count in level_counts.items():
        lines.append(f"- {level}: {count} Zeilen")
    lines.append(f"- Eindeutige Artikel-IDs: {pairing['unique_ids']}")
    lines.append(f"- Vollstaendige Paare (beide Niveaus vorhanden): {pairing['complete_pairs']}")
    if pairing["only_klexikon"]:
        lines.append(f"- NUR Klexikon vorhanden (MiniKlexikon fehlt): {pairing['only_klexikon']}")
    if pairing["only_mini"]:
        lines.append(f"- NUR MiniKlexikon vorhanden (Klexikon fehlt): {pairing['only_mini']}")
    lines.append("")

    # --- Duplikate ---
    lines.append("## Duplikate & Near-Duplikate")
    lines.append(f"- IDs mit mehr als 2 Vorkommen (ungewoehnlich, da normal 1x Klexikon + 1x MiniKlexikon): "
                 f"{len(exact_dup)}")
    if exact_dup:
        for id_, count in list(exact_dup.items())[:10]:
            lines.append(f"  - '{id_}': {count}x")
    lines.append(f"- Near-Duplicate-Gruppen (nur Gross-/Kleinschreibung/ß-ss unterschiedlich): {len(near_dup)}")
    for norm, variants in list(near_dup.items())[:10]:
        lines.append(f"  - {variants}")
    lines.append("")

    # --- Boilerplate & Markup ---
    n_boilerplate = sum(1 for m in metrics if m["has_boilerplate"])
    n_markup = sum(1 for m in metrics if m["has_markup_artifact"])
    lines.append("## Boilerplate & Markup-Reste")
    lines.append(f"- Targets mit MiniKlexikon-Navigationsfloskel ('Mehr über ... Frag Finn'): "
                 f"{n_boilerplate} ({n_boilerplate/len(metrics)*100:.1f}%)")
    lines.append(f"- Quellen mit erkennbaren Wiki-Markup-Resten: {n_markup} ({n_markup/len(metrics)*100:.1f}%)")
    lines.append("")

    # --- Laengen & Kompression ---
    lines.append("## Laenge & Kompression (gesamt)")
    lines.append(describe([m["source_words"] for m in metrics], "Quelle (Woerter)"))
    lines.append(describe([m["target_words"] for m in metrics], "Ziel (Woerter)"))
    lines.append(describe([m["compression_ratio"] for m in metrics], "Kompressionsrate (Ziel/Quelle)"))
    n_target_longer = sum(1 for m in metrics if (m["compression_ratio"] or 0) > 1)
    lines.append(f"Ziel laenger als Quelle: {n_target_longer} von {len(metrics)} "
                 f"({n_target_longer/len(metrics)*100:.1f}%)")
    lines.append("")

    lines.append("## Grounding-Score (Wortueberlappung Quelle<->Ziel)")
    lines.append(describe([m["grounding_score"] for m in metrics], "Grounding gesamt"))
    lines.append("Niedriger Wert = viel Zieltext ohne Beleg in der Quelle -> Halluzinationsrisiko beim Training.")
    lines.append("")

    lines.append("## Pro Niveau")
    for level, recs in by_level.items():
        lines.append(f"### {level} (n={len(recs)})")
        lines.append(describe([m["source_words"] for m in recs], "  Quelle (Woerter)"))
        lines.append(describe([m["target_words"] for m in recs], "  Ziel (Woerter)"))
        lines.append(describe([m["target_sentences"] for m in recs], "  Ziel (Saetze)"))
        lines.append(describe([m["grounding_score"] for m in recs], "  Grounding"))
        if _HAS_TEXTSTAT:
            lines.append(describe([m["target_wstf"] for m in recs], "  Ziel-WSTF"))
            lines.append(describe([m["target_lix"] for m in recs], "  Ziel-LIX"))
            lines.append(describe([m["target_flesch_amstad"] for m in recs], "  Ziel-Flesch-Amstad"))
        if any("target_mean_word_aoa" in m for m in recs):
            lines.append(describe([m.get("target_mean_word_aoa") for m in recs], "  Ziel mittlere Wort-AoA"))
            lines.append(describe([m.get("target_p90_word_aoa") for m in recs], "  Ziel p90 Wort-AoA"))
        lines.append("")

    # --- Split-Check ---
    if split_info:
        lines.append("## Split-Vollstaendigkeit (train/val/test)")
        lines.append(f"Gefundene Splits: {split_info['splits_found']}")
        for name, size in split_info["sizes"].items():
            lines.append(f"  {name}: {size} eindeutige IDs")
        if split_info["overlaps"]:
            lines.append("ACHTUNG -- ueberlappende IDs zwischen Splits (Data Leakage):")
            for pair, n in split_info["overlaps"].items():
                lines.append(f"  {pair}: {n} gemeinsame IDs")
        else:
            lines.append("Keine ID-Ueberlappung zwischen den Splits gefunden.")
        lines.append("")

    # --- Extremwerte ---
    lines.append("## Auffaellige Einzelfaelle")
    sorted_by_grounding = sorted(metrics, key=lambda m: m["grounding_score"])
    lines.append("Niedrigstes Grounding (5):")
    for m in sorted_by_grounding[:5]:
        lines.append(f"  - {m['id']} ({m['level']}): grounding={m['grounding_score']:.3f}")
    lines.append("Hoechste Kompressionsrate / laengstes Ziel relativ zur Quelle (5):")
    for m in sorted(metrics, key=lambda m: m["compression_ratio"] or 0, reverse=True)[:5]:
        lines.append(f"  - {m['id']} ({m['level']}): compression_ratio={m['compression_ratio']:.2f}")
    lines.append("")

    report_path = output_dir / "analysis_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("Report geschrieben nach %s", report_path)


def export_csv(output_dir, metrics):
    if not metrics:
        return
    path = output_dir / "per_record_metrics.csv"
    fieldnames = list(metrics[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)
    logger.info("Pro-Datensatz-Metriken exportiert nach %s", path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default="./dataset_mt5/all.jsonl")
    p.add_argument("--output-dir", default="./dataset-analysis")
    p.add_argument("--aoa-file", default="./lexical_ressources/GPT_estimates_AoA.xlsx",
                    help="Optional -- wird uebersprungen, falls Datei nicht existiert.")
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Lade %s ...", args.input)
    records = load_jsonl(args.input)
    logger.info("%d Zeilen geladen.", len(records))

    aoa_lookup = load_aoa_lookup(args.aoa_file)
    if aoa_lookup:
        logger.info("AoA-Ressource geladen: %d Woerter.", len(aoa_lookup))

    logger.info("Berechne Metriken pro Datensatz ...")
    metrics = [compute_record_metrics(r, aoa_lookup) for r in records]

    pairing = check_level_pairing(records)
    exact_dup, near_dup = find_duplicates(records)
    split_info = check_split_files(args.input)

    export_csv(output_dir, metrics)
    write_report(output_dir, records, metrics, pairing, exact_dup, near_dup, split_info, args)

    logger.info("Erzeuge Diagramme ...")
    by_level_words = defaultdict(list)
    by_level_grounding = defaultdict(list)
    by_level_lix = defaultdict(list)
    for m in metrics:
        by_level_words[m["level"]].append(m["target_words"])
        by_level_grounding[m["level"]].append(m["grounding_score"])
        by_level_lix[m["level"]].append(m.get("target_lix"))

    plot_histogram(None, "Zieltext-Laenge (Woerter) pro Niveau", "Woerter",
                    output_dir / "hist_target_words.png", by_level=by_level_words)
    plot_histogram(None, "Grounding-Score pro Niveau", "Grounding-Score",
                    output_dir / "hist_grounding.png", by_level=by_level_grounding)
    plot_histogram([m["compression_ratio"] for m in metrics], "Kompressionsrate (Ziel/Quelle, gesamt)",
                    "Kompressionsrate", output_dir / "hist_compression.png")
    if _HAS_TEXTSTAT:
        plot_histogram(None, "Ziel-LIX pro Niveau", "LIX",
                        output_dir / "hist_lix.png", by_level=by_level_lix)

    logger.info("Fertig. Ergebnisse in %s", output_dir.resolve())


if __name__ == "__main__":
    main()