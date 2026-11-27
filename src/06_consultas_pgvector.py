"""Actividad 5: consultas semánticas por similitud del coseno en pgvector (con filtros opcionales)."""
import argparse
import os
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

DSN = os.getenv(
    "PGVECTOR_DSN",
    "host=localhost port=5432 dbname=vectordb user=postgres password=postgres",
)
TABLA = "documentos_leyes"
MODELO = "intfloat/multilingual-e5-small"
EF_SEARCH = 64


def conectar():
    conn = psycopg.connect(DSN, autocommit=True)
    register_vector(conn)
    conn.execute(f"SET hnsw.ef_search = {EF_SEARCH}")
    try:
        conn.execute("SET hnsw.iterative_scan = relaxed_order")
    except Exception:
        pass
    return conn


def buscar_similares(conn, modelo, pregunta, top_k=5, categoria=None, documento=None):
    """Realiza búsqueda semántica por similitud del coseno con filtros opcionales."""
    vector_consulta = modelo.encode(
        f"query: {pregunta}", normalize_embeddings=True
    ).astype("float32")

    condiciones = []
    params = [vector_consulta]

    if categoria:
        condiciones.append("categoria = %s")
        params.append(categoria)
    if documento:
        condiciones.append("documento ILIKE %s")
        params.append(f"%{documento}%")

    where_sql = f"WHERE {' AND '.join(condiciones)}" if condiciones else ""
    sql = f"""
        SELECT
            id,
            1 - (vector <=> %s::vector) AS similitud,
            documento,
            categoria,
            articulo,
            pagina,
            contenido
        FROM {TABLA}
        {where_sql}
        ORDER BY vector <=> %s::vector
        LIMIT %s
    """
    params.extend([vector_consulta, top_k])

    t0 = time.perf_counter()
    filas = conn.execute(sql, params).fetchall()
    tiempo_ms = (time.perf_counter() - t0) * 1000
    return filas, tiempo_ms


def imprimir_resultados(pregunta, filas, tiempo_ms, categoria=None, documento=None):
    filtros = []
    if categoria:
        filtros.append(f"categoria='{categoria}'")
    if documento:
        filtros.append(f"documento~='{documento}'")
    etiqueta_filtro = f" | Filtros: {', '.join(filtros)}" if filtros else ""

    print(f"\nPregunta: {pregunta}{etiqueta_filtro}")
    print(f"Tiempo de consulta en pgvector: {tiempo_ms:.2f} ms | Resultados: {len(filas)}")
    print("-" * 80)
    for rank, (rid, sim, doc, cat, art, pag, texto) in enumerate(filas, start=1):
        resumen = texto.replace("\n", " ")[:220]
        print(f"{rank}. Similitud: {sim:.4f} | [{cat}] {doc} — Art. {art} (pág. {pag}, id {rid})")
        print(f"   {resumen}...\n")


def main():
    parser = argparse.ArgumentParser(
        description="Consultas semánticas por similitud del coseno en pgvector."
    )
    parser.add_argument("pregunta", nargs="?", help="Texto de la consulta en lenguaje natural")
    parser.add_argument("--k", type=int, default=5, help="Cantidad de resultados (Top-K)")
    parser.add_argument("--categoria", help="Filtrar por categoría exacta")
    parser.add_argument("--documento", help="Filtrar por nombre parcial del documento")
    args = parser.parse_args()

    modelo = SentenceTransformer(MODELO)
    conn = conectar()

    if args.pregunta:
        filas, ms = buscar_similares(
            conn, modelo, args.pregunta, top_k=args.k,
            categoria=args.categoria, documento=args.documento,
        )
        imprimir_resultados(args.pregunta, filas, ms, args.categoria, args.documento)
    else:
        # Demostración por defecto: consulta general + consulta con filtro de metadatos
        demos = [
            ("¿Cuántos días de vacaciones le corresponden a un trabajador?", 3, None, None),
            ("¿Qué pasa si me despiden sin justa causa?", 3, None, None),
            ("¿Qué equipo de protección personal debe entregar el empleador?", 3, "seguridad ocupacional", None),
        ]
        for preg, k, cat, doc in demos:
            filas, ms = buscar_similares(conn, modelo, preg, top_k=k, categoria=cat, documento=doc)
            imprimir_resultados(preg, filas, ms, cat, doc)

    conn.close()


if __name__ == "__main__":
    main()
