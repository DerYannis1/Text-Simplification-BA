### Evaluate the results on the Geolino dataset. (probably) ###
import json
from collections import defaultdict

entries = [json.loads(l) for l in open("./dataset/all.jsonl", encoding="utf-8")]

# Wie viele Artikel haben beide Levels?
by_id = defaultdict(set)
for e in entries:
    by_id[e["id"]].add(e["level"])

both_levels = [id_ for id_, levels in by_id.items() if len(levels) == 2]
only_klexikon = [id_ for id_, levels in by_id.items() if levels == {"Klexikon"}]
only_mini = [id_ for id_, levels in by_id.items() if levels == {"MiniKlexikon"}]

print(f"Artikel mit beiden Levels: {len(both_levels)}")
print(f"Nur Klexikon: {len(only_klexikon)}")
print(only_klexikon)
print(f"Nur MiniKlexikon: {len(only_mini)}")
print(f"Gesamt eindeutige Artikel: {len(by_id)}")


#Percentile for each similarity
import pandas as pd

df = pd.read_csv("./dataset/src_tar.csv")

percentiles = [0, 0.01, 0.05, 0.10, 0.25, 0.30, 0.75, 0.90, 0.95, 0.99, 1.0]

result = (
    df["similarity"]
    .quantile(percentiles)
    .rename_axis("Percentile")
    .reset_index(name="Similarity")
)

result["Percentile"] = (result["Percentile"] * 100).astype(int).astype(str) + "%"

print(result)


#Avg. Sentence Length per level

#Avg. Word length per level

#Avg. flesh score

#Avg. WSTF Score

