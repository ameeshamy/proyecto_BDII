import json
from collections import Counter
from pathlib import Path

chunks = json.loads(Path("data/chunks/chunks_limpios.json").read_text(encoding="utf-8"))

cont = Counter((c["archivo"][:30], c["articulo"]) for c in chunks)
print("=== Artículos con más chunks (partes) ===")
for (archivo, art), n in cont.most_common(10):
    print(f"{archivo} | art. {art}: {n} chunks")

print("\n=== Detalle de los artículos sospechosos ===")
for archivo_pref, art in [("DIRECTOR-DE-SALARIOS", 6), ("LEY DEL SERVICIO CIVIL", 72)]:
    print(f"\n{archivo_pref} art. {art}")
    for c in chunks:
        if c["archivo"].startswith(archivo_pref) and c["articulo"] == art:
            print(f"  id {c['id']} | parte {c['parte']} | pág {c['pagina']} | {c['n_palabras']} palabras | {c['contenido'][:90]}")