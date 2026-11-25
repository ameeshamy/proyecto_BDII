import json
import re
from bisect import bisect_left, bisect_right
from collections import defaultdict
from pathlib import Path

PROC = Path("data/processed")
OUT = Path("data/chunks")
OUT.mkdir(parents=True, exist_ok=True)

# Ajusta esto cuando elijan el modelo de embeddings: los modelos cortan el texto
# según su límite de TOKENS (y en español un token es menos que una palabra).
MAX_PALABRAS = 150
OVERLAP = 25
MAX_SALTO = 3  # cuántos números se pueden saltar (artículos que no aparecen) entre un artículo y el siguiente
MAX_RACHA = 40  # tope para recuperarse de una racha de artículos sin encabezado "limpio" (ej. varios Derogado seguidos)

# Reconoce: "ARTÍCULO 37.-", "Artículo 6.", "Artículo. 527.", "ARTICULO No. 4", "Artículo N.- 7"
ENCABEZADO = re.compile(
    r"\b(ART[ÍI]CULO|Art[íi]culo)\.?\s*(?:N[o°º]?[.\-]*\s*)?(\d{1,4})\b\s*[.\-–—°º]*"
)

# Donde termina la ley y empiezan firmas, fechas o anexos (decretos de reforma pegados al final)
FIN_LEY = re.compile(
    r"(?i:\bdado en\b|\bcomun[ií]quese\b|\bpublicado en\b)"
    r"|\bCONSIDERANDO\b|\bEL CONGRESO NACIONAL\b"
    r"|\bDECRETO\s+(?:N[ÚU]MERO|N[úu]mero|No\.?|N[°º])\s*\d"
)


# Cierres "fuertes": si aparecen entre dos artículos, la ley terminó y lo que sigue es un anexo
CIERRE_FUERTE = re.compile(
    r"\bDado en (?:la ciudad|el Sal[oó]n|el Palacio|Tegucigalpa)\b"
    r"|\bCONSIDERANDO\b|\bEL CONGRESO NACIONAL\b|\bDECRETO\s+N[ÚU]MERO\b"
)


# Documentos cuyos anexos (decretos de reforma, reglamento, tablas) SÍ se procesan:
# cada decreto anexo se trata como un documento aparte, con su propia numeración.
PROCESAR_ANEXOS = {"DIRECTOR-DE-SALARIOS-LEY-DEL-SALARIO-MINIMO.pdf"}
LIMITE_ANEXO = 110  # palabras por chunk en los anexos (más cortos, para no partir las tablas a la mitad)

# Encabezados de los decretos anexos: "DECRETO NUMERO 717 LA JUNTA...", "ACUERDO N. 12 Tegucigalpa..."
DECRETO = re.compile(
    r"\b(?:(DECRETO(?:-LEY)?(?:\s+EJECUTIVO)?)\s+(?:N[ÚU]MERO|N)[.\-]*\s*(\d{1,3}(?:-\d{2,3})?)"
    r"|(ACUERDO)\s+N\.\s*(\d{1,3}))\s+(?=EL\b|LA\b|La\b|Tegucigalpa)"
)


def es_encabezado(texto, pos, en_mayusculas):
    """¿Es un encabezado real de artículo y no una referencia ("conforme al Artículo 7")?
    Lo es si: está en MAYÚSCULAS, o viene después de punto/fin de oración, o después de un
    título en mayúsculas ("... OBLIGACIONES PATRONALES Artículo 12"), o después de un número."""
    if en_mayusculas:
        return True
    i = pos - 1
    while i >= 0 and texto[i].isspace():
        i -= 1
    if i < 0:
        return True
    if texto[i] in '.;:)]»"”' or texto[i].isdigit():
        return True
    if re.search(r"(?:derogad[oa]s?|zonas?)\W*$", texto[max(0, i - 15): i + 1], re.I):
        return True  # "Artículo 31. Derogado Artículo 32. ..." o el final de una tabla "... En todas las zonas Artículo 2."
    palabras = re.findall(r"[^\W\d_]+", texto[max(0, i - 30): i + 1])
    return bool(palabras) and palabras[-1].isupper()


def hueco_lleno(por_num, desde, hasta, a, b):
    """¿Existe algún candidato (de cualquier tipo) para cada número entre a+1 y b-1, en orden,
    entre las posiciones desde y hasta?"""
    ult = desde
    for n in range(a + 1, b):
        posiciones = por_num.get(n, [])
        k = bisect_right(posiciones, ult)
        if k >= len(posiciones) or posiciones[k] >= hasta:
            return False
        ult = posiciones[k]
    return True


def secuencia(cands, texto, reiniciar=True):
    """Recorre el texto de izquierda a derecha y encadena 1, 2, 3...
    - Solo cuenta encabezados reales (no referencias).
    - Si un número no aparece, permite saltar hasta MAX_SALTO, pero no si el número que falta
      sí aparece un poco más adelante (entonces el candidato era una referencia).
    - Si hay una racha larga de artículos que no parecen encabezados (ej. varios derogados seguidos),
      puede saltar más, siempre que todos los números intermedios existan en el hueco.
    - Si la numeración vuelve a empezar en 1 (solo si reiniciar=True), o aparece el cierre de la ley,
      lo que sigue es un anexo y se detiene."""
    por_num = defaultdict(list)
    for c in cands:
        por_num[c[1]].append(c[0])

    camino, actual = [], 0
    for i, (pos, num, es_enc) in enumerate(cands):
        if not es_enc:
            continue
        if reiniciar and num == 1 and actual >= 3:
            break
        if camino and CIERRE_FUERTE.search(texto, camino[-1][0], pos):
            break
        if actual < num <= actual + MAX_SALTO:
            if num > actual + 1:
                falta = any(
                    c[2] and actual < c[1] < num and c[0] - pos < 40000 for c in cands[i + 1:i + 40]
                )
                if falta:
                    continue
        elif not (
            camino
            and actual + MAX_SALTO < num <= actual + MAX_RACHA
            and hueco_lleno(por_num, camino[-1][0], pos, actual, num)
        ):
            continue
        camino.append((pos, num, es_enc))
        actual = num
    return rellenar(camino, cands)


def rellenar(camino, cands):
    """Si faltan números en la cadena (por ejemplo, un encabezado que no venía después de un punto),
    se buscan entre los candidatos descartados, solo en el hueco donde deberían estar."""
    resultado = []
    for item in camino:
        if resultado:
            ant_pos, ant_num = resultado[-1][0], resultado[-1][1]
            for falta in range(ant_num + 1, item[1]):
                for c in cands:
                    if c[1] == falta and ant_pos < c[0] < item[0]:
                        resultado.append(c)
                        ant_pos = c[0]
                        break
        resultado.append(item)
    return resultado


def trocear(texto):
    """Parte un artículo largo en chunks de hasta MAX_PALABRAS palabras, cortando al final de una
    oración o inciso (después de . ; :) y repitiendo al inicio del siguiente las últimas oraciones
    (hasta OVERLAP palabras) para no perder el hilo."""
    if len(texto.split()) <= MAX_PALABRAS:
        return [texto]
    paso = MAX_PALABRAS - OVERLAP
    piezas = []
    for s in (x for x in re.split(r"(?<=[.;:])\s+", texto) if x.strip()):
        w = s.split()
        if len(w) > paso:  # un tramo largo sin puntuación (una tabla, por ejemplo): se corta por palabras
            piezas.extend(" ".join(w[i:i + paso]) for i in range(0, len(w), paso))
        else:
            piezas.append(s)
    chunks, actual, nuevas, n = [], [], [], 0
    for s in piezas:
        w = len(s.split())
        if actual and n + w > MAX_PALABRAS:
            chunks.append(" ".join(actual))
            cola, m = [], 0
            for x in reversed(actual):  # solapamiento: las últimas oraciones que quepan en OVERLAP
                wx = len(x.split())
                if m + wx > OVERLAP:
                    break
                cola.insert(0, x)
                m += wx
            actual, nuevas, n = cola, [], m
        actual.append(s)
        nuevas.append(s)
        n += w
    if actual:
        # si lo que sobra es muy corto (un título suelto, una línea), se pega al chunk anterior
        if chunks and sum(len(x.split()) for x in nuevas) < 15:
            chunks[-1] += " " + " ".join(nuevas)
        else:
            chunks.append(" ".join(actual))
    return chunks


def limpiar_id(s):
    """'40-891' -> '40-89': el último dígito es el número de una nota al pie pegado al decreto."""
    m = re.fullmatch(r"(\d{1,3})-(\d{3})", s)
    return f"{m.group(1)}-{m.group(2)[:2]}" if m else s


UNIDADES = {"uno": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5, "seis": 6, "siete": 7, "ocho": 8, "nueve": 9}
DECENAS = {"sesenta": 60, "setenta": 70, "ochenta": 80, "noventa": 90}


def anio_decreto(seg, id_decreto, es_acuerdo):
    """Año del decreto: del propio número ('30-93' -> 1993), de la frase 'Dado en ... de mil novecientos
    setenta y ocho', o del primer año con dígitos (para el Acuerdo)."""
    m = re.fullmatch(r"\d{1,3}-(\d{2})", id_decreto)
    if m:
        return f"19{m.group(1)}"
    if es_acuerdo:
        y = re.search(r"\b(19[5-9]\d)\b", seg)
        return y.group(1) if y else "s. f."
    k = seg.rfind("Dado en")
    if k >= 0:
        m = re.search(r"mil novecientos\s+((?:[a-záéíóú]+\s*){1,3})", seg[k:], re.I)
        if m:
            palabras = m.group(1).lower().split()
            dec = next((DECENAS[w] for w in palabras if w in DECENAS), 0)
            uni = next((UNIDADES[w] for w in palabras if w in UNIDADES), 0)
            if dec:
                return str(1900 + dec + uni)
    y = re.findall(r"\b(19[5-9]\d)\b", seg)
    return y[-1] if y else "s. f."


def etiqueta_anexo(m, texto_decreto):
    if m.group(3):
        num = m.group(4)
        return f"Acuerdo {num} (Reglamento del Decreto-Ley 121, {anio_decreto(texto_decreto, num, True)})"
    num = limpiar_id(m.group(2))
    return f"{m.group(1).title()} {num} ({anio_decreto(texto_decreto, num, False)})"


def cortes_filas(texto):
    """Puntos donde se puede cortar sin partir una fila de una tabla: fin de oración, o justo
    antes del nombre de una actividad (una palabra con mayúscula que viene después de un número)."""
    puntos = {0, len(texto)}
    for m in re.finditer(r"(?<=[.;])\s+(?=[A-ZÁÉÍÓÚ\"“(])", texto):
        puntos.add(m.end())
    for m in re.finditer(
        r"(?<=\d)\s+(?=(?!En\b|De\b|Trabajadores\b|Tabla\b)(?:[A-ZÁÉÍÓÚ][a-záéíóúñ]|[A-ZÁÉÍÓÚ]{3,}))", texto
    ):
        puntos.add(m.end())
    for m in re.finditer(r"\b[Zz]onas?\s+(?=[A-ZÁÉÍÓÚ])", texto):
        puntos.add(m.end())
    return sorted(puntos)


def trocear_anexo(texto):
    puntos = cortes_filas(texto)
    segmentos = [texto[a:b].strip() for a, b in zip(puntos, puntos[1:]) if texto[a:b].strip()]
    chunks, actual, n = [], [], 0
    for s in segmentos:
        w = len(s.split())
        if w > LIMITE_ANEXO:
            if actual:
                chunks.append(" ".join(actual))
                actual, n = [], 0
            chunks.extend(trocear(s))
            continue
        if n + w > LIMITE_ANEXO and actual:
            chunks.append(" ".join(actual))
            actual, n = [], 0
        actual.append(s)
        n += w
    if actual:
        chunks.append(" ".join(actual))
    return chunks


def es_tabla(texto):
    """Chunk con cifras tipo 13.80 o 9.25 (valores de una tabla de salarios)."""
    toks = texto.split()
    cifras = sum(1 for t in toks if re.search(r"\d[.,]\d{2}\b", t))
    return cifras >= 2 or (cifras >= 1 and cifras / len(toks) >= 0.03)


def columnas(texto_decreto):
    if re.search(r"\bZONAS?\b", texto_decreto):
        return "columnas de la tabla: Zona I, Zona II, Zona III"
    if re.search(r"D\.\s?C\.\s?(?:/|y)\s?S\.\s?P\.\s?S\.", texto_decreto):
        return "columnas de la tabla: Distrito Central y San Pedro Sula, Resto del país"
    return ""


def chunks_de_anexos(anexo, base, d, paginas, inicios):
    """Divide el anexo en decretos y chunkea cada uno. Devuelve (chunks, resumen, texto_antes_del_primero)."""
    ms = list(DECRETO.finditer(anexo))
    if not ms:
        return [], [], anexo
    chunks, resumen = [], []
    for k, m in enumerate(ms):
        fin_seg = ms[k + 1].start() if k + 1 < len(ms) else len(anexo)
        seg = anexo[m.start():fin_seg]
        etiqueta = etiqueta_anexo(m, seg)
        cols = columnas(seg)
        cands = [
            (x.start(), int(x.group(2)), es_encabezado(seg, x.start(), x.group(1).isupper()))
            for x in ENCABEZADO.finditer(seg)
        ]
        camino = secuencia(cands, seg, reiniciar=False)
        n_antes = len(chunks)
        for j, (ini, num, _) in enumerate(camino):
            if j + 1 < len(camino):
                fin = camino[j + 1][0]
            else:
                f = FIN_LEY.search(seg, ini + 20)
                fin = f.start() if f else len(seg)
            contenido = seg[ini:fin].strip()
            pagina = paginas[bisect_right(inicios, base + m.start() + ini) - 1]["pagina"]
            for parte, trozo in enumerate(trocear_anexo(contenido), start=1):
                pref = f"[{d['documento']}, anexo: {etiqueta}"
                pref += f"; {cols}]" if cols and es_tabla(trozo) else "]"
                texto_chunk = f"{pref} {trozo}"
                chunks.append({
                    "documento": f"{d['documento']} (anexo: {etiqueta})",
                    "categoria": d["categoria"],
                    "fuente": d["fuente"],
                    "archivo": d["archivo"],
                    "articulo": num,
                    "pagina": pagina,
                    "parte": parte,
                    "contenido": texto_chunk,
                    "n_palabras": len(texto_chunk.split()),
                })
        resumen.append((etiqueta, len(camino), len(chunks) - n_antes))
    return chunks, resumen, anexo[:ms[0].start()]


def chunks_de_documento(d):
    paginas = d["paginas"]
    partes, inicios, pos = [], [], 0
    for p in paginas:
        inicios.append(pos)
        partes.append(p["texto"])
        pos += len(p["texto"]) + 1
    texto = " ".join(partes)

    cands = [
        (m.start(), int(m.group(2)), es_encabezado(texto, m.start(), m.group(1).isupper()))
        for m in ENCABEZADO.finditer(texto)
    ]
    camino = secuencia(cands, texto)

    chunks, descartado, largo_ultimo = [], "", 0
    for k, (ini, num, _) in enumerate(camino):
        if k + 1 < len(camino):
            fin = camino[k + 1][0]
        else:  # último artículo: termina donde empiezan las firmas, la fecha o los anexos
            m = FIN_LEY.search(texto, ini + 20)
            fin = m.start() if m else len(texto)
            cola = texto[fin:]
            descartado = cola.strip()
            base_cola = fin + (len(cola) - len(cola.lstrip()))
            largo_ultimo = fin - ini
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

    resumen_anexos = []
    if descartado and d["archivo"] in PROCESAR_ANEXOS:
        extra, resumen_anexos, descartado = chunks_de_anexos(descartado, base_cola, d, paginas, inicios)
        chunks.extend(extra)

    nums = [c[1] for c in camino]
    faltan = sorted(set(range(nums[0], nums[-1] + 1)) - set(nums)) if nums else []
    aviso = ""
    if camino and largo_ultimo > 6000:
        mayores = sorted({c[1] for c in cands if c[0] > camino[-1][0] and camino[-1][1] < c[1] <= camino[-1][1] + 60})
        if mayores:
            aviso = f"la numeración se detuvo en el art. {camino[-1][1]}, pero más adelante hay encabezados con números mayores (ej.: {mayores[:8]})"
    return chunks, len(nums), faltan, (nums[0], nums[-1]) if nums else None, descartado, largo_ultimo, aviso, resumen_anexos


if __name__ == "__main__":
    todos, id_global = [], 1
    for f in sorted(PROC.glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        chunks, n_art, faltan, rango, descartado, largo_ultimo, aviso, resumen_anexos = chunks_de_documento(d)
        for c in chunks:
            c["id"] = id_global
            id_global += 1
        todos.extend(chunks)

        principal = [c for c in chunks if c["documento"] == d["documento"]]
        pals = [c["n_palabras"] for c in principal] or [0]
        print(f"\n{d['archivo']}")
        print(f"  artículos detectados: {n_art} (del {rango[0]} al {rango[1]})" if rango else "  ⚠ no se detectó ningún artículo")
        print(f"  chunks de la ley: {len(principal)} | palabras: mín {min(pals)}, máx {max(pals)}, prom {sum(pals)//len(pals)}")
        if faltan:
            print(f"  ⚠ números sin detectar ({len(faltan)}): {faltan[:20]}{' ...' if len(faltan) > 20 else ''}")

        partes_por_art = {}
        for c in principal:
            partes_por_art[c["articulo"]] = partes_por_art.get(c["articulo"], 0) + 1
        top = sorted(partes_por_art.items(), key=lambda x: -x[1])[:3]
        print("  artículos con más partes: " + ", ".join(f"art. {a} ({n})" for a, n in top))

        for etiqueta, n_art_a, n_chunks_a in resumen_anexos:
            print(f"  anexo {etiqueta}: {n_art_a} artículos, {n_chunks_a} chunks")
        if descartado:
            (OUT / f"{f.stem}_texto_descartado.txt").write_text(descartado, encoding="utf-8")
            print(f"  texto después del último artículo (firmas/anexos) descartado: {len(descartado):,} caracteres")
        if largo_ultimo > 6000:
            print(f"  ⚠ el último artículo mide {largo_ultimo:,} caracteres: revisar si arrastra anexos")
        if aviso:
            print(f"  ⚠ {aviso}")

    (OUT / "chunks.json").write_text(json.dumps(todos, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nTotal de chunks: {len(todos)} -> {OUT / 'chunks.json'}")