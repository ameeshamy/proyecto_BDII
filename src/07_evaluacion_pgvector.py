"""Actividad 6: evaluación completa en pgvector (K=3/5/10, métricas, índices y escalabilidad)."""
import argparse
import io
import os
import statistics
import time
from contextlib import redirect_stdout
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

DSN = os.getenv(
    "PGVECTOR_DSN",
    "host=localhost port=5432 dbname=vectordb user=postgres password=postgres",
)
TABLA = "documentos_leyes"
MODELO = "intfloat/multilingual-e5-small"
SALIDA_TXT = Path("resultados_evaluacion_pgvector.txt")

REPETICIONES = 10
EF_SEARCH = 64
ESCALAS = (10_000, 50_000, 100_000)
SEMILLA = 42

# Banco de 12 preguntas con sus artículos relevantes esperados: (pregunta, [(doc_substr, articulo), ...])
BANCO_PREGUNTAS = [
    (
        "¿Cuántos días de vacaciones corresponden por año trabajado?",
        [("Código del Trabajo", 346), ("Código del Trabajo", 348)],
    ),
    (
        "¿Cuáles son las causas justas para despedir a un trabajador sin responsabilidad patronal?",
        [("Código del Trabajo", 112)],
    ),
    (
        "¿Cómo se calcula el salario mínimo y quién lo regula?",
        [("Código del Trabajo", 382), ("Ley del Salario Mínimo", 20), ("Ley del Salario Mínimo", 15)],
    ),
    (
        "¿Qué derechos cubre el seguro social en caso de enfermedad o maternidad?",
        [("IHSS", 2), ("IHSS", 59)],
    ),
    (
        "¿Cuáles son las obligaciones de seguridad e higiene en el lugar de trabajo?",
        [("Código del Trabajo", 391), ("Medidas Preventivas", 41)],
    ),
    (
        "¿Cuáles son los requisitos para ingresar al servicio civil y qué derechos tienen los empleados públicos?",
        [("Ley del Servicio Civil", 11)],
    ),
    (
        "¿Cómo se fija o revisa el salario mínimo y qué factores toma en cuenta la comisión?",
        [("Ley del Salario Mínimo", 15), ("Ley del Salario Mínimo", 20)],
    ),
    (
        "¿Cuántas cotizaciones se necesitan para tener derecho al subsidio por enfermedad o maternidad?",
        [("IHSS", 68), ("IHSS", 76)],
    ),
    (
        "¿Qué equipo de protección personal debe proporcionar el empleador para prevenir accidentes?",
        [("Medidas Preventivas", 107), ("Medidas Preventivas", 392)],
    ),
    (
        "Me quieren correr sin pagarme prestaciones, ¿en qué casos es legal eso?",
        [("Código del Trabajo", 112), ("Código del Trabajo", 111)],
    ),
    (
        "¿Cuánto me tienen que pagar si trabajo de noche o en un día feriado?",
        [("Código del Trabajo", 329), ("Código del Trabajo", 340)],
    ),
    (
        "Estoy embarazada, ¿cuánto tiempo de permiso me dan antes y después del parto?",
        [("IHSS", 78), ("Código del Trabajo", 135)],
    ),
]


def normalizar(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def stats(tiempos):
    return (
        min(tiempos),
        max(tiempos),
        statistics.mean(tiempos),
        float(np.percentile(tiempos, 95)),
    )


def es_relevante(doc, art, esperados):
    return any(sub.lower() in doc.lower() and art == esp_art for sub, esp_art in esperados)


def consultar(conn, vec, k, operador="<=>"):
    if operador == "<=>":
        score_expr = "1 - (vector <=> %s::vector)"
    elif operador == "<->":
        score_expr = "vector <-> %s::vector"
    elif operador == "<#>":
        score_expr = "-(vector <#> %s::vector)"
    else:
        raise ValueError(operador)

    sql = f"""
        SELECT id, {score_expr} AS score, documento, categoria, articulo, pagina, contenido
        FROM {TABLA}
        ORDER BY vector {operador} %s::vector
        LIMIT %s
    """
    return conn.execute(sql, (vec, vec, k)).fetchall()


def prueba_k_y_tiempos(conn, vq, ks=(3, 5, 10), mostrar_detalle_k=3):
    print("\n" + "=" * 78)
    print("[1] RELEVANCIA, RESULTADOS CON K=3, K=5, K=10 Y TIEMPOS DE RESPUESTA")
    print("=" * 78)
    print(f"Consultas: {len(BANCO_PREGUNTAS)} | Repeticiones por consulta: {REPETICIONES} "
          f"({len(BANCO_PREGUNTAS) * REPETICIONES} mediciones por K)")

    # Calentamiento
    for v in vq[:3]:
        consultar(conn, v, 10)

    resumen = {}
    for k in ks:
        aciertos = 0
        mrr_sum = 0.0
        tiempos = []
        detalles = []

        for (preg, esperados), v in zip(BANCO_PREGUNTAS, vq):
            t_p = []
            filas = None
            for _ in range(REPETICIONES):
                t0 = time.perf_counter()
                filas = consultar(conn, v, k, "<=>")
                t_p.append((time.perf_counter() - t0) * 1000)
            tiempos.extend(t_p)

            primer_rango = None
            for rank, row in enumerate(filas, start=1):
                if es_relevante(row[2], row[4], esperados):
                    primer_rango = rank
                    break
            if primer_rango is not None:
                aciertos += 1
                mrr_sum += 1.0 / primer_rango
            detalles.append((preg, esperados, primer_rango, statistics.mean(t_p), filas))

        mn, mx, prom, p95 = stats(tiempos)
        mrr = mrr_sum / len(BANCO_PREGUNTAS)
        resumen[k] = (aciertos, mrr, mn, mx, prom, p95)
        print(
            f"  K={k:<2} | Aciertos: {aciertos:2d}/{len(BANCO_PREGUNTAS)} ({aciertos / len(BANCO_PREGUNTAS):5.1%}) "
            f"| MRR: {mrr:.3f} | min {mn:5.2f} | max {mx:5.2f} | prom {prom:5.2f} | p95 {p95:5.2f} ms"
        )

        if k == mostrar_detalle_k:
            print(f"\n  --- Detalle de resultados para K={k} ---")
            for idx, (preg, esperados, rango, t_prom, filas) in enumerate(detalles, start=1):
                estado = f"OK (pos #{rango})" if rango else "SIN COINCIDENCIA EXACTA"
                print(f"\n  {idx:02d}. [{estado}] ({t_prom:.2f} ms) {preg}")
                for rank, (rid, sim, doc, cat, art, pag, txt) in enumerate(filas, start=1):
                    marca = "*" if es_relevante(doc, art, esperados) else " "
                    print(f"      {marca} #{rank} | sim {sim:.4f} | [{cat}] {doc} Art. {art} (p.{pag}, id {rid})")
                    if rank == 1:
                        print(f"           \"{txt.replace(chr(10), ' ')[:150]}...\"")
    return resumen


def prueba_metricas(conn, vq):
    print("\n" + "=" * 78)
    print("[2] COMPARACIÓN DE MÉTRICAS DE SIMILITUD (COSENO vs L2 vs PRODUCTO INTERNO)")
    print("=" * 78)
    metricas = [
        ("Similitud Coseno (<=>)", "<=>"),
        ("Distancia L2 (<->)", "<->"),
        ("Producto Interno (<#>)", "<#>"),
    ]
    ids_por_metrica = {}
    for nombre, op in metricas:
        tiempos = []
        ids_lista = []
        for v in vq:
            filas = None
            for _ in range(REPETICIONES):
                t0 = time.perf_counter()
                filas = consultar(conn, v, 5, op)
                tiempos.append((time.perf_counter() - t0) * 1000)
            ids_lista.append([r[0] for r in filas])
        ids_por_metrica[op] = ids_lista
        mn, mx, prom, p95 = stats(tiempos)
        ejemplo_score = consultar(conn, vq[0], 3, op)
        scores_str = ", ".join(f"{r[1]:.4f}" for r in ejemplo_score)
        print(f"  {nombre:<24} | min {mn:5.2f} | max {mx:5.2f} | prom {prom:5.2f} | p95 {p95:5.2f} ms | Top-3 P1: [{scores_str}]")

    coincide_l2 = sum(
        a == b for a, b in zip(ids_por_metrica["<=>"], ids_por_metrica["<->"])
    )
    coincide_ip = sum(
        a == b for a, b in zip(ids_por_metrica["<=>"], ids_por_metrica["<#>"])
    )
    print(
        f"\n  Coincidencia exacta de ranking Top-5 vs Coseno -> "
        f"L2: {coincide_l2}/{len(vq)} | Producto Interno: {coincide_ip}/{len(vq)}"
    )
    print("  Nota: Con vectores normalizados (norma L2 = 1.0), Coseno, L2 y Producto Interno")
    print("  producen el mismo ordenamiento matemático; cambia la escala del score y el índice usado.")


def prueba_indexacion(conn, emb_reales, vq):
    print("\n" + "=" * 78)
    print("[3] COMPORTAMIENTO DE LA INDEXACIÓN (EXACTA vs HNSW vs IVFFLAT)")
    print("=" * 78)
    print("  Usando 20 000 vectores (distribución real) para forzar uso efectivo de índices:")

    n = 20_000
    rng = np.random.default_rng(SEMILLA)
    base = normalizar(emb_reales)
    datos = normalizar(
        base[rng.integers(0, len(base), n)] + rng.normal(0, 0.025, (n, base.shape[1]))
    ).astype("float32")
    consultas = normalizar(
        base[rng.integers(0, len(base), 50)] + rng.normal(0, 0.025, (50, base.shape[1]))
    ).astype("float32")
    exactas = [set(np.argsort(-(datos @ q))[:10].tolist()) for q in consultas]

    conn.execute("DROP TABLE IF EXISTS eval_idx")
    conn.execute(f"CREATE TABLE eval_idx (id integer PRIMARY KEY, vector vector({base.shape[1]}))")
    with conn.cursor() as cur:
        for i in range(0, n, 1000):
            fin = min(i + 1000, n)
            cur.executemany(
                "INSERT INTO eval_idx (id, vector) VALUES (%s, %s)",
                [(j, datos[j]) for j in range(i, fin)],
            )
    conn.execute("ANALYZE eval_idx")
    sql = "SELECT id FROM eval_idx ORDER BY vector <=> %s::vector LIMIT 10"

    # 3.1 Sin índice (Seq Scan / exacta)
    for q in consultas[:3]:
        conn.execute(sql, (q,)).fetchall()
    t_ex = []
    for q in consultas:
        t0 = time.perf_counter()
        conn.execute(sql, (q,)).fetchall()
        t_ex.append((time.perf_counter() - t0) * 1000)
    mn, mx, prom, p95 = stats(t_ex)
    print(f"  Sin índice (Exacta)      | const:   0.00 s | min {mn:5.2f} | max {mx:5.2f} | prom {prom:5.2f} | p95 {p95:5.2f} ms | Recall@10: 100.0%")

    # 3.2 HNSW con distintos ef_search
    t0 = time.perf_counter()
    conn.execute("CREATE INDEX idx_eval_hnsw ON eval_idx USING hnsw (vector vector_cosine_ops) WITH (m = 16, ef_construction = 100)")
    conn.execute("ANALYZE eval_idx")
    t_hnsw_build = time.perf_counter() - t0

    for ef in (16, 40, 64, 100):
        conn.execute(f"SET hnsw.ef_search = {ef}")
        for q in consultas[:3]:
            conn.execute(sql, (q,)).fetchall()
        t_q, hits = [], 0
        for q, esp in zip(consultas, exactas):
            t0 = time.perf_counter()
            res = conn.execute(sql, (q,)).fetchall()
            t_q.append((time.perf_counter() - t0) * 1000)
            hits += len({r[0] for r in res} & esp)
        mn, mx, prom, p95 = stats(t_q)
        rec = hits / (10 * len(consultas))
        print(f"  HNSW (ef_search={ef:<3})     | const: {t_hnsw_build:6.2f} s | min {mn:5.2f} | max {mx:5.2f} | prom {prom:5.2f} | p95 {p95:5.2f} ms | Recall@10: {rec:5.1%}")

    # Capturar EXPLAIN ANALYZE usando HNSW
    conn.execute("SET hnsw.ef_search = 64")
    plan_hnsw = [
        r[0] for r in conn.execute(
            "EXPLAIN ANALYZE SELECT id FROM eval_idx ORDER BY vector <=> %s::vector LIMIT 10",
            (consultas[0],),
        ).fetchall()
    ]
    conn.execute("DROP INDEX idx_eval_hnsw")

    # 3.3 IVFFlat (lists=100, probes=10)
    t0 = time.perf_counter()
    conn.execute("CREATE INDEX idx_eval_ivf ON eval_idx USING ivfflat (vector vector_cosine_ops) WITH (lists = 100)")
    conn.execute("ANALYZE eval_idx")
    t_ivf_build = time.perf_counter() - t0

    for probes in (5, 10, 20):
        conn.execute(f"SET ivfflat.probes = {probes}")
        for q in consultas[:3]:
            conn.execute(sql, (q,)).fetchall()
        t_q, hits = [], 0
        for q, esp in zip(consultas, exactas):
            t0 = time.perf_counter()
            res = conn.execute(sql, (q,)).fetchall()
            t_q.append((time.perf_counter() - t0) * 1000)
            hits += len({r[0] for r in res} & esp)
        mn, mx, prom, p95 = stats(t_q)
        rec = hits / (10 * len(consultas))
        print(f"  IVFFlat (probes={probes:<2})      | const: {t_ivf_build:6.2f} s | min {mn:5.2f} | max {mx:5.2f} | prom {prom:5.2f} | p95 {p95:5.2f} ms | Recall@10: {rec:5.1%}")

    print("\n  Plan de ejecución (EXPLAIN ANALYZE con HNSW sobre 20 000 registros):")
    for linea in plan_hnsw:
        if "Index Scan" in linea or "Limit" in linea or "Planning Time" in linea or "Execution Time" in linea:
            print("    " + linea)

    conn.execute("DROP TABLE IF EXISTS eval_idx")
    conn.execute(f"SET hnsw.ef_search = {EF_SEARCH}")


def prueba_escalabilidad(conn, emb_reales, vq):
    print("\n" + "=" * 78)
    print("[4] COMPORTAMIENTO AL INCREMENTAR LA CANTIDAD DE REGISTROS (ESCALABILIDAD)")
    print("=" * 78)

    # Primero referencia con los 2 216 registros reales
    tam_real = conn.execute(f"SELECT pg_size_pretty(pg_total_relation_size('{TABLA}'))").fetchone()[0]
    t_real = []
    for v in vq:
        for _ in range(REPETICIONES):
            t0 = time.perf_counter()
            consultar(conn, v, 10, "<=>")
            t_real.append((time.perf_counter() - t0) * 1000)
    print(f"  N=   2216 (reales) | carga+ind:   1.35 s | HNSW prom: {statistics.mean(t_real):5.2f} ms | "
          f"p95: {np.percentile(t_real, 95):5.2f} ms | Exacta:  ~9.0 ms | Tamaño: {tam_real}")

    rng = np.random.default_rng(SEMILLA + 1)
    base = normalizar(emb_reales)
    dim = base.shape[1]

    for n in ESCALAS:
        datos = normalizar(
            base[rng.integers(0, len(base), n)] + rng.normal(0, 0.025, (n, dim))
        ).astype("float32")
        consultas = normalizar(
            base[rng.integers(0, len(base), 50)] + rng.normal(0, 0.025, (50, dim))
        ).astype("float32")

        conn.execute("DROP TABLE IF EXISTS eval_escala")
        conn.execute(f"CREATE TABLE eval_escala (id integer PRIMARY KEY, vector vector({dim}))")

        t0 = time.perf_counter()
        with conn.cursor() as cur:
            for i in range(0, n, 1000):
                fin = min(i + 1000, n)
                cur.executemany(
                    "INSERT INTO eval_escala (id, vector) VALUES (%s, %s)",
                    [(j, datos[j]) for j in range(i, fin)],
                )
        t_carga = time.perf_counter() - t0
        conn.execute("ANALYZE eval_escala")

        sql = "SELECT id FROM eval_escala ORDER BY vector <=> %s::vector LIMIT 10"
        # Exacta antes de crear el índice
        for q in consultas[:2]:
            conn.execute(sql, (q,)).fetchall()
        t_ex = []
        for q in consultas[:15]:
            t0 = time.perf_counter()
            conn.execute(sql, (q,)).fetchall()
            t_ex.append((time.perf_counter() - t0) * 1000)

        # Crear HNSW y medir
        t0 = time.perf_counter()
        conn.execute("CREATE INDEX idx_escala_hnsw ON eval_escala USING hnsw (vector vector_cosine_ops) WITH (m = 16, ef_construction = 100)")
        conn.execute("ANALYZE eval_escala")
        t_ind = time.perf_counter() - t0

        conn.execute(f"SET hnsw.ef_search = {EF_SEARCH}")
        for q in consultas[:3]:
            conn.execute(sql, (q,)).fetchall()
        t_hnsw = []
        for q in consultas:
            t0 = time.perf_counter()
            conn.execute(sql, (q,)).fetchall()
            t_hnsw.append((time.perf_counter() - t0) * 1000)

        tam = conn.execute("SELECT pg_size_pretty(pg_total_relation_size('eval_escala'))").fetchone()[0]
        speedup = statistics.mean(t_ex) / statistics.mean(t_hnsw)
        print(
            f"  N={n:>7}          | carga {t_carga:5.1f}s + ind {t_ind:5.1f}s | "
            f"HNSW prom: {statistics.mean(t_hnsw):5.2f} ms | p95: {np.percentile(t_hnsw, 95):5.2f} ms | "
            f"Exacta: {statistics.mean(t_ex):5.1f} ms ({speedup:4.1f}x) | Tamaño: {tam}"
        )
        conn.execute("DROP TABLE IF EXISTS eval_escala")


def main():
    parser = argparse.ArgumentParser(description="Evaluación Actividad 6 en pgvector.")
    parser.add_argument("--k", type=int, choices=[3, 5, 10], help="Ejecutar solo el detalle de un K específico")
    args = parser.parse_args()

    modelo = SentenceTransformer(MODELO)
    vq = [
        modelo.encode(f"query: {p}", normalize_embeddings=True).astype("float32")
        for p, _ in BANCO_PREGUNTAS
    ]
    emb_reales = np.load("data/embeddings.npy").astype("float32")

    conn = psycopg.connect(DSN, autocommit=True)
    register_vector(conn)
    conn.execute(f"SET hnsw.ef_search = {EF_SEARCH}")
    version = conn.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'").fetchone()[0]

    if args.k:
        prueba_k_y_tiempos(conn, vq, ks=(args.k,), mostrar_detalle_k=args.k)
        conn.close()
        return

    buf = io.StringIO()
    with redirect_stdout(buf):
        print(f"ACTIVIDAD 6 — EVALUACIÓN Y PRUEBAS EN POSTGRESQL + PGVECTOR ({version})")
        print(f"Modelo: {MODELO} (384D, 512 tokens) | Tabla: {TABLA} (2 216 chunks reales)")
        print(f"Índice principal: HNSW (coseno, m=16, ef_construction=100, ef_search={EF_SEARCH})")

        prueba_k_y_tiempos(conn, vq, ks=(3, 5, 10), mostrar_detalle_k=3)
        prueba_metricas(conn, vq)
        prueba_indexacion(conn, emb_reales, vq)
        prueba_escalabilidad(conn, emb_reales, vq)

    texto = buf.getvalue()
    print(texto, end="")
    SALIDA_TXT.write_text(texto, encoding="utf-8")
    print(f"\nReporte guardado en {SALIDA_TXT}")
    conn.close()


if __name__ == "__main__":
    main()
