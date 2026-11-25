import json
import subprocess
import sys
from pathlib import Path

RUTA = "data/chunks/chunks_limpios.json"

# Versión guardada en un commit anterior de Git (por defecto HEAD) vs la que tienes ahora en la carpeta
ref = sys.argv[1] if len(sys.argv) > 1 else "HEAD"
salida = subprocess.run(["git", "show", f"{ref}:{RUTA}"], capture_output=True, encoding="utf-8")
if salida.returncode != 0:
    raise SystemExit(f"No encontré chunks_limpios.json en el commit {ref}. Mira tus commits con: git log --oneline")
print(f"Comparando contra el commit {ref}")
viejo = json.loads(salida.stdout)
nuevo = json.loads(Path(RUTA).read_text(encoding="utf-8"))

for doc in dict.fromkeys(c["documento"] for c in nuevo):
    sv = {c["seccion"] for c in viejo if c["documento"] == doc and c["seccion"]}
    sn = {c["seccion"] for c in nuevo if c["documento"] == doc and c["seccion"]}
    perdidas, nuevas = sv - sn, sn - sv
    if not perdidas and not nuevas:
        continue
    print(f"\n{doc}: antes {len(sv)} secciones, ahora {len(sn)}")
    for s in sorted(perdidas):
        arts = sorted({c["articulo"] for c in viejo if c["documento"] == doc and c["seccion"] == s})
        print(f"  SE PERDIÓ: {s[:90]}\n             (antes en los artículos {arts[:6]}{'...' if len(arts) > 6 else ''})")
    for s in sorted(nuevas):
        arts = sorted({c["articulo"] for c in nuevo if c["documento"] == doc and c["seccion"] == s})
        print(f"  NUEVA:     {s[:90]}\n             (ahora en los artículos {arts[:6]}{'...' if len(arts) > 6 else ''})")
print("\nListo.")