import json
import re
from bisect import bisect_right
from pathlib import Path

PROC = Path("data/processed")
OUT = Path("data/chunks")
OUT.mkdir(parents=True, exist_ok=True)

# Ajustar al momento de elegir el modelo de embeddings
MAX_PALABRAS = 150
OVERLAP = 25

# Reconoce: "ARTÍCULO 37.-", "Artículo 6.", "Artículo. 527.", "ARTICULO No. 4"
ENCABEZADO = re.compile(
    r"\b(ART[ÍI]CULO|Art[íi]culo)\.?\s*(?:N[o°º]\.?\s*)?(\d{1,4})\b\s*[.\-–—°º]*"
)


def inicia_oracion(texto, pos):
    i = pos - 1
    while i >= 0 and texto[i].isspace():
        i -= 1
    return i < 0 or texto[i] in '.;:)]»"”'


def secuencia_valida(cands):
    """De todos los candidatos, se queda con la secuencia creciente más larga.
    Desempata a favor de los que parecen encabezados de verdad."""
    n = len(cands)
    if n == 0:
        return []
    mejor = [(1, 1 if c[2] else 0) for c in cands]
    prev = [-1] * n
    for i in range(n):
        for j in range(i):
            if cands[j][1] < cands[i][1]:
                nuevo = (mejor[j][0] + 1, mejor[j][1] + (1 if cands[i][2] else 0))
                if nuevo > mejor[i]:
                    mejor[i], prev[i] = nuevo, j
    i = max(range(n), key=lambda k: mejor[k])
    camino = []
    while i != -1:
        camino.append(cands[i])
        i = prev[i]
    return camino[::-1]


def trocear(texto):
    palabras = texto.split()
    if len(palabras) <= MAX_PALABRAS:
        return [texto]
    paso = MAX_PALABRAS - OVERLAP
    partes = []
    for ini in range(0, len(palabras), paso):
        partes.append(" ".join(palabras[ini:ini + MAX_PALABRAS]))
        if ini + MAX_PALABRAS >= len(palabras):
            break
    return partes


def chunks_de_documento(d):
    paginas = d["paginas"]
    partes, inicios, pos = [], [], 0
    for p in paginas:
        inicios.append(pos)
        partes.append(p["texto"])
        pos += len(p["texto"]) + 1
    texto = " ".join(partes)

    cands = [
        (m.start(), int(m.group(2)), m.group(1).isupper() or inicia_oracion(texto, m.start()))
        for m in ENCABEZADO.finditer(texto)
    ]
    camino = secuencia_valida(cands)

    chunks = []
    for k, (ini, num, _) in enumerate(camino):
        fin = camino[k + 1][0] if k + 1 < len(camino) else len(texto)
        contenido = texto[ini:fin].strip()
        pagina = paginas[bisect_right(inicios, ini) - 1]["pagina"]
        for parte, trozo in enumerate(trocear(contenido), start=1):
            if parte > 1:
                trozo = f"Artículo {num} (continuación): {trozo}"
            chunks.append({
                "documento": d["documento"],
                "categoria": d["categoria"],
                "fuente": d["fuente"],
                "archivo": d["archivo"],
                "articulo": num,
                "pagina": pagina,
                "parte": parte,
                "contenido": trozo,
                "n_palabras": len(trozo.split()),
            })

    nums = [c[1] for c in camino]
    faltan = sorted(set(range(nums[0], nums[-1] + 1)) - set(nums)) if nums else []
    return chunks, len(nums), faltan, (nums[0], nums[-1]) if nums else None


if __name__ == "__main__":
    todos, id_global = [], 1
    for f in sorted(PROC.glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        chunks, n_art, faltan, rango = chunks_de_documento(d)
        for c in chunks:
            c["id"] = id_global
            id_global += 1
        todos.extend(chunks)

        pals = [c["n_palabras"] for c in chunks] or [0]
        print(f"\n{d['archivo']}")
        print(f"  artículos detectados: {n_art} (del {rango[0]} al {rango[1]})" if rango else "  ⚠ no se detectó ningún artículo")
        print(f"  chunks: {len(chunks)} | palabras: mín {min(pals)}, máx {max(pals)}, prom {sum(pals)//len(pals)}")
        if faltan:
            print(f"  ⚠ números sin detectar ({len(faltan)}): {faltan[:20]}{' ...' if len(faltan) > 20 else ''}")

    (OUT / "chunks.json").write_text(json.dumps(todos, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nTotal de chunks: {len(todos)} -> {OUT / 'chunks.json'}")