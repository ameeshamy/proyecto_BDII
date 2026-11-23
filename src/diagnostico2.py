import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location("chunking", "src/02_chunking.py")
ch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ch)

PRUEBAS = [
    ("DIRECTOR-DE-SALARIOS-LEY-DEL-SALARIO-MINIMO", [6, 7, 8, 12]),
    ("LEY DEL SERVICIO CIVIL", [70, 71, 72]),
]

for nombre, numeros in PRUEBAS:
    d = json.loads(Path(f"data/processed/{nombre}.json").read_text(encoding="utf-8"))
    print(f"\n=== {nombre} ({len(d['paginas'])} págs) ===")
    for p in d["paginas"]:
        for m in ch.ENCABEZADO.finditer(p["texto"]):
            num = int(m.group(2))
            if num in numeros:
                ctx = p["texto"][max(0, m.start() - 30): m.end() + 30]
                print(f"  pág {p['pagina']} | art. {num} | ...{ctx!r}")