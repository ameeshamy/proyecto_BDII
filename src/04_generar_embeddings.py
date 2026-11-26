"""
Actividad 3 — Generación de embeddings para los chunks de leyes laborales.

Modelo seleccionado: intfloat/multilingual-e5-small
- Dimensionalidad: 384
- Límite de contexto: 512 tokens (0% de recorte en nuestros 2 216 chunks)
- Entrenado para recuperación semántica asimétrica:
    * Documentos/artículos llevan prefijo "passage: "
    * Consultas del usuario llevan prefijo "query: "
"""
import json
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

MODELO = "intfloat/multilingual-e5-small"
ENTRADA = Path("data/chunks/chunks_limpios.json")
SALIDA = Path("data/embeddings.npy")


def main():
    chunks = json.loads(ENTRADA.read_text(encoding="utf-8"))
    # Los modelos E5 requieren el prefijo "passage: " para indexar documentos
    textos = [f"passage: {c['contenido']}" for c in chunks]

    print(f"Cargando modelo: {MODELO}")
    modelo = SentenceTransformer(MODELO)
    max_tokens = modelo.max_seq_length

    # Verificación de longitud en tokens para garantizar que no se recorta texto
    longitudes = np.array([
        len(modelo.tokenizer.encode(t, truncation=False)) for t in textos
    ])
    recortados = int((longitudes > max_tokens).sum())
    print(f"Límite del modelo: {max_tokens} tokens")
    print(
        f"Tokens por chunk -> mediana: {int(np.median(longitudes))} | "
        f"p95: {int(np.percentile(longitudes, 95))} | "
        f"máximo: {int(longitudes.max())} | recortados: {recortados}/{len(chunks)}"
    )

    t0 = time.perf_counter()
    embeddings = modelo.encode(
        textos,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype("float32")
    elapsed = time.perf_counter() - t0

    np.save(SALIDA, embeddings)
    print(f"\nGuardado en: {SALIDA}")
    print(f"Tiempo de generación: {elapsed:.2f} s ({len(chunks) / elapsed:.1f} chunks/s)")
    print(f"Forma de la matriz: {embeddings.shape} | Dtype: {embeddings.dtype}")
    print(f"Dimensionalidad: {embeddings.shape[1]}")
    print(f"Norma L2 del vector 0: {np.linalg.norm(embeddings[0]):.4f}")
    print(f"Ejemplo (primeros 8 valores del chunk 1): {np.round(embeddings[0][:8], 5)}")


if __name__ == "__main__":
    main()
