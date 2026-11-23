import json
import re
from pathlib import Path

chunks = json.loads(Path("data/chunks/chunks.json").read_text(encoding="utf-8"))

print("=== Chunks más cortos (menos de 12 palabras) ===")
for c in chunks:
    if c["n_palabras"] < 12:
        print(f"[{c['archivo'][:25]}] art. {c['articulo']}: {c['contenido']}")

print("\n=== Chunks con posible basura de símbolos ===")
for c in chunks:
    txt = c["contenido"]
    simbolos = sum(1 for ch in txt if not ch.isalnum() and not ch.isspace() and ch not in ".,;:()-–—\"'“”«»¿?¡!%$/°º")
    if simbolos > 5 or re.search(r"\*\+|\*!!", txt):
        print(f"[{c['archivo'][:25]}] art. {c['articulo']}: {txt[:150]}")

print("\n=== Final de cada documento (últimos 300 caracteres del último chunk) ===")
vistos = {}
for c in chunks:
    vistos[c["archivo"]] = c
for archivo, c in vistos.items():
    print(f"\n[{archivo[:40]}] art. {c['articulo']}:\n...{c['contenido'][-300:]}")

print("\n=== Contexto de los artículos 526-530 del Código ===")
for c in chunks:
    if c["archivo"] == "codigo_de_trabajo.pdf" and 525 <= c["articulo"] <= 530:
        print(f"art. {c['articulo']}: {c['contenido'][:200]}\n")