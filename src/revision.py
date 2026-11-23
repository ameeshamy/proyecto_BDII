import json, re
from pathlib import Path

patron = re.compile(r"art[ií]culo\s+(?:no\.?\s*)?\d+", re.I)
for f in sorted(Path("data/processed").glob("*.json")):
    d = json.loads(f.read_text(encoding="utf-8"))
    texto = " ".join(p["texto"] for p in d["paginas"])
    print(f"{d['archivo']}: {len(patron.findall(texto))} menciones de 'Artículo N'")