import json
import re
from pathlib import Path

# Títulos de capítulo/sección que quedaron DENTRO del texto de un chunk (sin separar como sección)
TITULO = re.compile(
    r"(?:T[ÍI]TULO|CAP[ÍI]TULO|LIBRO|SECCI[ÓO]N)\s+"
    r"(?:[IVXLCDM]+|\d+|[ÚU]NICO|PRIMERO|SEGUNDO|TERCERO|CUARTO|QUINTO)\b"
)
chunks = json.loads(Path("data/chunks/chunks_limpios.json").read_text(encoding="utf-8"))

n = 0
for c in chunks:
    for m in TITULO.finditer(c["contenido"]):
        n += 1
        ini = max(0, m.start() - 70)
        print(f"[{c['documento'][:45]}] art. {c['articulo']} parte {c['parte']} | id {c['id']}")
        print(f"   ...{c['contenido'][ini:m.end() + 70]}...\n")
print(f"{n} títulos sueltos encontrados.")