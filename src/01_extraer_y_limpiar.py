import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pymupdf

RAW = Path("data/raw")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

DOCUMENTOS = {
    "codigo_de_trabajo.pdf": {
        "documento": "Código del Trabajo",
        "categoria": "contrato de trabajo",
        "fuente": "PENDIENTE: URL y fecha de consulta",
    },
    "DIRECTOR-DE-SALARIOS-LEY-DEL-SALARIO-MINIMO.pdf": {
        "documento": "Ley del Salario Mínimo",
        "categoria": "salarios",
        "fuente": "PENDIENTE: URL y fecha de consulta",
    },
    "LEY DEL SERVICIO CIVIL.pdf": {
        "documento": "Ley del Servicio Civil",
        "categoria": "empleo público",
        "fuente": "PENDIENTE: URL y fecha de consulta",
    },
    "reglamento de aplicacion de la ley del ihss.pdf": {
        "documento": "Reglamento de Aplicación de la Ley del IHSS",
        "categoria": "seguridad social",
        "fuente": "PENDIENTE: URL y fecha de consulta",
    },
    "Reglamento-General-de-Medidas-Preventivas-de-Accidentes-de-Trabajo-FINAL-1.pdf": {
        "documento": "Reglamento General de Medidas Preventivas de Accidentes de Trabajo",
        "categoria": "seguridad ocupacional",
        "fuente": "PENDIENTE: URL y fecha de consulta",
    },
}

# Documentos con notas al pie: se descarta la letra más pequeña que (proporción × tamaño normal del texto).
FILTRAR_NOTAS = {
    "codigo_de_trabajo.pdf": 0.9,
    "LEY DEL SERVICIO CIVIL.pdf": 0.9,
}

# Línea que es solo un número de página: "12", "Página 12", "12 de 154"
NUM_PAGINA = re.compile(r"(p[aá]gina\s*)?\d{1,3}(\s*(de|/)\s*\d{1,3})?", re.I)

# Tramos de 4 o más "palabras" hechas solo de símbolos
SIMBOLOS = re.compile(r"(?:(?<=\s)|^)(?:[^\w\s]{1,4}\s+){3,}[^\w\s]{1,4}(?=\s|$)")


def clave(linea):
    """Normaliza una línea para detectar encabezados repetidos (los números cuentan igual)."""
    return re.sub(r"\d+", "#", linea.strip().lower())


def detectar_repetidas(paginas_lineas, proporcion=0.3):
    """Líneas del borde de la página (primeras/últimas 2) que se repiten en muchas páginas."""
    cont = Counter()
    for lineas in paginas_lineas:
        lineas = [l.strip() for l in lineas if l.strip()]
        for k in {clave(l) for l in lineas[:2] + lineas[-2:]}:
            cont[k] += 1
    minimo = max(3, proporcion * len(paginas_lineas))
    return {k for k, v in cont.items() if v >= minimo and not k.startswith("art")}


def extraer_lineas(pdf, ratio):
    """Devuelve (líneas por página, texto descartado por ser letra pequeña, tamaño base)."""
    if not ratio:
        return [p.get_text("text", sort=True).splitlines() for p in pdf], [], None

    paginas = []
    for pagina in pdf:
        bloques = [b for b in pagina.get_text("dict")["blocks"] if b["type"] == 0]
        bloques.sort(key=lambda b: (round(b["bbox"][1]), b["bbox"][0]))
        paginas.append(bloques)

    tam = Counter()
    for bloques in paginas:
        for b in bloques:
            for l in b["lines"]:
                for s in l["spans"]:
                    tam[round(s["size"], 1)] += len(s["text"].strip())
    base = tam.most_common(1)[0][0]  # el tamaño que más se usa = texto normal

    resultado, descartado = [], []
    for bloques in paginas:
        lineas = []
        for b in bloques:
            for l in b["lines"]:
                partes = []
                for s in l["spans"]:
                    if s["size"] < base * ratio:
                        descartado.append(s["text"])
                    else:
                        partes.append(s["text"])
                linea = "".join(partes)
                if linea.strip():
                    lineas.append(linea)
        resultado.append(lineas)
    return resultado, descartado, base


def limpiar_pagina(lineas, repetidas, eliminadas):
    lineas = [l.strip() for l in lineas if l.strip()]
    conservadas = []
    for i, l in enumerate(lineas):
        en_borde = i < 2 or i >= len(lineas) - 2
        if en_borde and (clave(l) in repetidas or NUM_PAGINA.fullmatch(l)):
            eliminadas.append(l)
            continue
        if not any(c.isalnum() for c in l):  # línea sin letras ni números
            eliminadas.append(l)
            continue
        conservadas.append(l)

    texto = "\n".join(conservadas)
    texto = unicodedata.normalize("NFC", texto)
    texto = re.sub(r"([a-záéíóúñ])-\n([a-záéíóúñ])", r"\1\2", texto)  # palabras partidas por guion
    texto = re.sub(r"\s*\n\s*", " ", texto)  # une líneas en un solo párrafo
    texto = SIMBOLOS.sub(" ", texto)  # basura inline
    texto = re.sub(r"\s{2,}", " ", texto).strip()
    return texto


def procesar(pdf_path):
    doc = pymupdf.open(pdf_path)
    paginas_lineas, notas, base = extraer_lineas(doc, FILTRAR_NOTAS.get(pdf_path.name))
    if notas:
        (OUT / f"{pdf_path.stem}_notas.txt").write_text("\n".join(notas), encoding="utf-8")
        print(f"   tamaño base {base} pt: se descartaron {len(notas)} fragmentos de letra pequeña")
    repetidas = detectar_repetidas(paginas_lineas)

    eliminadas, paginas = [], []
    for n, lineas in enumerate(paginas_lineas, start=1):
        paginas.append({"pagina": n, "texto": limpiar_pagina(lineas, repetidas, eliminadas)})

    meta = DOCUMENTOS.get(
        pdf_path.name,
        {"documento": pdf_path.stem, "categoria": "PENDIENTE", "fuente": "PENDIENTE"},
    )
    salida = {**meta, "archivo": pdf_path.name, "paginas": paginas}

    (OUT / f"{pdf_path.stem}.json").write_text(
        json.dumps(salida, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT / f"{pdf_path.stem}.txt").write_text(
        "\n\n".join(f"[Página {p['pagina']}]\n{p['texto']}" for p in paginas), encoding="utf-8"
    )
    # Reporte de lo que se borró, para revisarlo y confirmar que no se fue nada importante
    reporte = Counter(eliminadas).most_common()
    (OUT / f"{pdf_path.stem}_eliminado.txt").write_text(
        "\n".join(f"{v}x  {k}" for k, v in reporte), encoding="utf-8"
    )

    vacias = [p["pagina"] for p in paginas if len(p["texto"]) < 50]
    total = sum(len(p["texto"]) for p in paginas)
    print(f"{pdf_path.name}: {len(paginas)} págs, {total:,} caracteres, "
          f"{len(eliminadas)} líneas eliminadas")
    if vacias:
        print(f"   ⚠ Páginas casi vacías (¿escaneadas o en blanco?): {vacias}")


if __name__ == "__main__":
    pdfs = sorted(RAW.glob("*.pdf"))
    for pdf in pdfs:
        if pdf.name not in DOCUMENTOS:
            print(f"⚠ {pdf.name} NO está en DOCUMENTOS (categoria y fuente quedarán en PENDIENTE)")
    for pdf in pdfs:
        procesar(pdf)