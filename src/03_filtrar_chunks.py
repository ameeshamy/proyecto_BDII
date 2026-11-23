import json
import re
from collections import Counter
from pathlib import Path

ENTRADA = Path("data/chunks/chunks.json")
SALIDA = Path("data/chunks/chunks_limpios.json")

PREFIJO = re.compile(r"^Art[ií]culo\.?\s+\d+(?:\s*\(continuaci[oó]n\))?\s*[.:\-–]*\s*", re.I)
PUNTOS = re.compile(r"[.…·_]{4,}")  # puntos guía de las tablas: ………………
CIERRE = re.compile(r"\b(?:dado en|comun[ií]quese|publicado en)\b", re.I)
FIRMAS = re.compile(r"\s--\s?[A-ZÁÉÍÓÚÑ]{3,}") 
ENCABEZADO = re.compile(
    r"\s((?:T[ÍI]TULO|CAP[ÍI]TULO|LIBRO|SECCI[ÓO]N)\s+"
    r"(?:[IVXLCDM]+|\d+|[ÚU]NICO|PRIMERO|SEGUNDO|TERCERO|CUARTO|QUINTO)\b.*)$",
    re.S,
)


def separar_encabezado(texto):
    """Si al final del chunk quedó pegado el título/capítulo que viene después, lo separa.
    Si detrás del encabezado hay texto de un artículo, ese texto se CONSERVA en el chunk."""
    m = ENCABEZADO.search(texto, max(0, len(texto) - 250))
    if not m:
        return texto, None, False
    antes, cola = texto[:m.start()].rstrip(), m.group(1)
    corte = re.search(r"\bart[ií]culo\b", cola, re.I)
    if corte:
        encabezado = cola[:corte.start()].strip()
        resto = cola[corte.start():].strip()
        return f"{antes} {resto}", encabezado, True
    return antes, cola.strip(), False


chunks = json.loads(ENTRADA.read_text(encoding="utf-8"))

ultimo_art = {}
for c in chunks:
    ultimo_art[c["archivo"]] = max(ultimo_art.get(c["archivo"], 0), c["articulo"])

limpios, stats = [], Counter()
archivo_actual, seccion_actual = None, ""

for c in chunks:
    if c["archivo"] != archivo_actual:
        archivo_actual, seccion_actual = c["archivo"], ""

    texto = c["contenido"]

    # 1. Puntos guía de tablas
    nuevo = PUNTOS.sub(" ", texto)
    if nuevo != texto:
        stats["chunks con puntos guía limpiados"] += 1
    texto = re.sub(r"\s{2,}", " ", nuevo).strip()

    # 2. Título/capítulo pegado al final: se saca del texto y pasa a ser la sección de los siguientes
    texto, encabezado, con_resto = separar_encabezado(texto)
    seccion_chunk = seccion_actual
    if encabezado:
        seccion_actual = encabezado
        stats["encabezados separados"] += 1
    if con_resto:
        stats["encabezados con texto de artículo detrás (revisar)"] += 1
        print(f"   revisar: {c['archivo'][:25]} art. {c['articulo']}: ...{texto[-150:]}")

    # 3. Artículos derogados
    cuerpo = PREFIJO.sub("", texto)
    if re.match(r"derogad[oa]", cuerpo, re.I) and len(cuerpo.split()) < 20:
        stats["chunks derogados descartados"] += 1
        continue

    # 4. Cierre del documento (firmas, fecha, publicación en La Gaceta) en el último artículo
    if c["articulo"] == ultimo_art[c["archivo"]]:
        pos = [m.start() for m in (CIERRE.search(texto), FIRMAS.search(texto)) if m]
        if pos:
            texto = texto[:min(pos)].rstrip()
            stats["cierres recortados"] += 1
            if len(PREFIJO.sub("", texto).split()) < 5:
                stats["chunks de cierre descartados"] += 1
                continue

    c["contenido"] = texto
    c["n_palabras"] = len(texto.split())
    c["seccion"] = seccion_chunk
    limpios.append(c)

for i, c in enumerate(limpios, start=1):
    c["id"] = i

SALIDA.write_text(json.dumps(limpios, ensure_ascii=False, indent=2), encoding="utf-8")

print(f"Chunks antes: {len(chunks)} -> después: {len(limpios)}")
for k, v in stats.items():
    print(f"  {k}: {v}")

print("\n=== Por documento ===")
for archivo in dict.fromkeys(c["archivo"] for c in limpios):
    grupo = [c for c in limpios if c["archivo"] == archivo]
    secciones = list(dict.fromkeys(c["seccion"] for c in grupo if c["seccion"]))
    print(f"\n{archivo}: {len(grupo)} chunks, {len(secciones)} secciones distintas")
    for s in secciones[:4]:
        print(f"   - {s[:90]}")
    print(f"   final del último chunk: ...{grupo[-1]['contenido'][-200:]}")

print("\n=== Chunks que siguen con menos de 10 palabras ===")
for c in limpios:
    if c["n_palabras"] < 10:
        print(f"[{c['archivo'][:20]}] art. {c['articulo']}: {c['contenido']}")