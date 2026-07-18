### Evaluate the results on the Geolino dataset. (probably) ###
import json
from collections import defaultdict

entries = [json.loads(l) for l in open("./dataset/train.jsonl", encoding="utf-8")]

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