"""Print a quick summary of the latest C++ analysis results."""
from pathlib import Path

f = Path("data/multilanguage_analysis/cpp_concept_matches.csv")
if not f.exists():
    print("No results yet. Run 'just analyse-cpp' first.")
else:
    rows = f.read_text().splitlines()[1:]
    files = {r.split(";")[0] for r in rows if r}
    types: dict[str, int] = {}
    patterns: dict[str, int] = {}
    for r in rows:
        if not r:
            continue
        parts = r.split(";")
        types[parts[3]] = types.get(parts[3], 0) + 1
        patterns[parts[2]] = patterns.get(parts[2], 0) + 1
    print(f"Total matches : {len(rows)}")
    print(f"Unique files  : {len(files)}")
    print(f"Match types   : {types}")
    print("Patterns found:")
    for p, c in sorted(patterns.items(), key=lambda x: -x[1]):
        print(f"  {c:3d}  {p}")
