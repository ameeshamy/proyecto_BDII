"""Actividad 4 y 5: crea la tabla documentos_leyes, carga vectores e indexa con HNSW (coseno)."""
import json
import os
import time
from pathlib import Path

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

DSN = os.getenv(
    "PGVECTOR_DSN",
    "host=localhost port=5432 dbname=vectordb user=postgres password=postgres",
)
TABLA = "documentos_leyes"
CHUNKS_PATH = Path("data/chunks/chunks_limpios.json")
EMBEDDINGS_PATH = Path("data/embeddings.npy")

# Parámetros del índice HNSW
HNSW_M = 16
HNSW_EF_CONSTRUCTION = 100


def main():
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    embeddings = np.load(EMBEDDINGS_PATH).astype("float32")
    assert len(chunks) == len(embeddings), (
        f"No coinciden chunks ({len(chunks)}) y embeddings ({len(embeddings)})"
    )
    dim = int(embeddings.shape[1])

    conn = psycopg.connect(DSN, autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)

    version = conn.execute(
        "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
    ).fetchone()[0]
    print(f"Conectado a PostgreSQL con pgvector {version}")

    # 1. Crear estructura de almacenamiento
    conn.execute(f"DROP TABLE IF EXISTS {TABLA}")
    conn.execute(f"""
        CREATE TABLE {TABLA} (
            id         INTEGER PRIMARY KEY,
            vector     vector({dim}) NOT NULL,
            contenido  TEXT NOT NULL,
            documento  TEXT NOT NULL,
            categoria  TEXT NOT NULL,
            fuente     TEXT NOT NULL,
            articulo   INTEGER,
            pagina     INTEGER,
            parte      INTEGER,
            seccion    TEXT
        )
    """)
    print(f"Tabla '{TABLA}' creada (dimensión del vector: {dim}).")

    # 2. Cargar los registros en lotes
    filas = [
        (
            int(c["id"]),
            embeddings[i],
            c["contenido"],
            c["documento"],
            c["categoria"],
            c["fuente"],
            c.get("articulo"),
            c.get("pagina"),
            c.get("parte"),
            c.get("seccion", ""),
        )
        for i, c in enumerate(chunks)
    ]

    t0 = time.perf_counter()
    with conn.cursor() as cur:
        for i in range(0, len(filas), 500):
            cur.executemany(
                f"""
                INSERT INTO {TABLA}
                    (id, vector, contenido, documento, categoria, fuente, articulo, pagina, parte, seccion)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                filas[i:i + 500],
            )
    t_carga = time.perf_counter() - t0
    print(f"Carga completada: {len(filas)} registros en {t_carga:.2f} s")

    # 3. Crear índices (vectorial HNSW para coseno + metadatos)
    t0 = time.perf_counter()
    conn.execute(f"""
        CREATE INDEX idx_{TABLA}_vector_hnsw
        ON {TABLA}
        USING hnsw (vector vector_cosine_ops)
        WITH (m = {HNSW_M}, ef_construction = {HNSW_EF_CONSTRUCTION})
    """)
    t_hnsw = time.perf_counter() - t0

    conn.execute(f"CREATE INDEX idx_{TABLA}_categoria ON {TABLA} (categoria)")
    conn.execute(f"CREATE INDEX idx_{TABLA}_documento ON {TABLA} (documento)")
    conn.execute(f"ANALYZE {TABLA}")
    print(
        f"Índice HNSW (coseno, m={HNSW_M}, ef_construction={HNSW_EF_CONSTRUCTION}) "
        f"creado en {t_hnsw:.2f} s"
    )

    # 4. Verificación y resumen
    total = conn.execute(f"SELECT count(*) FROM {TABLA}").fetchone()[0]
    tam_tabla, tam_total = conn.execute(
        f"SELECT pg_size_pretty(pg_relation_size('{TABLA}')), "
        f"pg_size_pretty(pg_total_relation_size('{TABLA}'))"
    ).fetchone()

    print(f"\nRegistros en '{TABLA}': {total}")
    print(f"Tamaño en disco -> tabla: {tam_tabla} | total con índices: {tam_total}")

    print("\nDistribución por categoría:")
    for cat, cnt in conn.execute(
        f"SELECT categoria, count(*) FROM {TABLA} GROUP BY categoria ORDER BY count(*) DESC"
    ).fetchall():
        print(f"  - {cat}: {cnt}")

    conn.close()


if __name__ == "__main__":
    main()
