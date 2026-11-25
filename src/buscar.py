import json
import math
import re
import sys
import unicodedata
from pathlib import Path

PARADAS = set("""
de del la las el los lo un una unos unas y e o u a al en con por para sin sobre entre que se su sus mi mis tu
es son fue ser esta este estos estas como cual cuales quien quienes cuando cuanto cuanta cuantos donde puede
pueden debe deben hay tiene tienen mas muy ya no si le les me nos hacer cada todo toda todos todas
""".split())


def norm(t):
    """Minúsculas y sin tildes, para que 'despido' encuentre 'DESPIDO' y 'Despído'."""
    t = unicodedata.normalize("NFD", t.lower())
    return "".join(c for c in t if unicodedata.category(c) != "Mn")


largo = 220  # cuántos caracteres de cada chunk se muestran en la lista
if "--largo" in sys.argv:
    k = sys.argv.index("--largo")
    largo = int(sys.argv[k + 1])
    del sys.argv[k:k + 2]

if len(sys.argv) < 2:
    print('Uso: python src/buscar.py "palabras clave" ["parte del nombre del documento"] [--largo 600]')
    print('     python src/buscar.py --id 123                       (muestra el chunk completo)')
    print('     python src/buscar.py --articulo 37 "medidas"        (muestra TODO el artículo, todas sus partes)')
    sys.exit()

chunks = json.loads(Path("data/chunks/chunks_limpios.json").read_text(encoding="utf-8"))

if sys.argv[1] == "--id":
    for c in chunks:
        if c["id"] == int(sys.argv[2]):
            print(f"[{c['documento']}] art. {c['articulo']} | pág {c['pagina']} | id {c['id']}\n")
            print(c["contenido"])
    sys.exit()

if sys.argv[1] == "--articulo":
    filtro = norm(sys.argv[3]) if len(sys.argv) > 3 else ""
    for c in chunks:
        if c["articulo"] == int(sys.argv[2]) and filtro in norm(c["documento"]):
            print(f"[{c['documento']}] art. {c['articulo']} | parte {c['parte']} | pág {c['pagina']} | id {c['id']}\n")
            print(c["contenido"] + "\n")
    sys.exit()

palabras = [p for p in re.findall(r"[a-z0-9]+", norm(sys.argv[1])) if p not in PARADAS and len(p) >= 3]
if not palabras:
    print("No quedaron palabras útiles. Prueba con 2 a 4 palabras que probablemente use la ley.")
    sys.exit()
# Se compara por el inicio de la palabra (5 letras), así 'informar' encuentra 'informarán'
prefijos = list(dict.fromkeys(p[:5] for p in palabras))
minimo = 1 if len(prefijos) <= 2 else math.ceil(len(prefijos) / 2)
filtro = norm(sys.argv[2]) if len(sys.argv) > 2 else ""

resultados = []
for c in chunks:
    if filtro and filtro not in norm(c["documento"]):
        continue
    t = norm(c["contenido"])
    presentes = [p for p in prefijos if re.search(r"\b" + re.escape(p), t)]
    if len(presentes) >= minimo:
        total = sum(len(re.findall(r"\b" + re.escape(p), t)) for p in presentes)
        resultados.append((len(presentes), total, c))
resultados.sort(key=lambda r: (-r[0], -r[1] / max(1, r[2]["n_palabras"])))  # primero los que tienen más palabras distintas; luego los más 'densos'

print(f"Palabras usadas: {', '.join(palabras)}  (mínimo {minimo} por chunk)")
print(f"{len(resultados)} chunks encontrados. Los 10 mejores:\n")
for n, _, c in resultados[:10]:
    print(f"[{c['documento'][:70]}] art. {c['articulo']} | pág {c['pagina']} | id {c['id']} | {n}/{len(prefijos)} palabras")
    print(f"   {c['contenido'][:largo]}\n")