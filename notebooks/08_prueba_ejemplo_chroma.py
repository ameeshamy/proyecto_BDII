"""Pruebas comparativas en Chroma (20 frases bancarias + 20 000 vectores) -> resultados_chroma.txt."""
import io
import os
import statistics
import time
from contextlib import redirect_stdout
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

RAIZ = Path(__file__).resolve().parent.parent
SALIDA_TXT = RAIZ / "resultados_chroma.txt"

PUERTO = 8001
REPETICIONES = 10
N_BENCH = 20000
SEMILLA = 42
M, EF_CONSTRUCTION, EF_SEARCH = 16, 100, 64

documentos = [
    "El cliente reporta que su tarjeta de débito fue rechazada en un cajero automático",
    "Solicitud de aumento de línea de crédito para cuenta empresarial",
    "El cliente no reconoce un cargo de $150 en su estado de cuenta",
    "Consulta sobre requisitos para abrir una cuenta de ahorros infantil",
    "Reclamo por doble cobro en transferencia interbancaria",
    "Solicitud de copia de estado de cuenta de los últimos 6 meses",
    "El cajero automático retuvo la tarjeta del cliente",
    "Consulta sobre tasa de interés para préstamo hipotecario",
    "Cliente reporta fraude: transacciones no reconocidas en su tarjeta de crédito",
    "Solicitud para cambiar el número de teléfono asociado a la cuenta",
    "El cliente reporta una transacción que no ha realizado con su tarjeta de crédito",
    "No está funcionando las transferencias ACH a las 08:00 am",
    "El usuario de banca en línea fue bloqueado y no puede entrar",
    "La app móvil se encuentra fuera de servicio",
    "El cliente solicita un préstam por consolidación de deudas",
    "La empresa ABC quiere cancelar su cuenta de cheques y abrir una nueva",
    "Se han incrementado las solicitudes de tarjeta de crédto para clientes con historial crediticio limitado",
    "Consulta para cancelar una tarjeta de crédito que ha sido robada",
    "El usuario de la app no puede ver el saldo de la tarjeta de crédito",
    "Solicitud para préstamo de pequeña empresa para capital de trabajo",
]

categorias = [
    "tarjetas", "prestamos", "fraude", "cuentas", "transferencias",
    "cuentas", "tarjetas", "prestamos", "fraude", "cuentas",
    "fraude", "canales_digitales", "canales_digitales", "canales_digitales", "prestamos",
    "cuentas", "tarjetas", "tarjetas", "canales_digitales", "prestamos",
]

preguntas_prueba = [
    ("alguien uso mi tarjeta sin permiso", "Cliente reporta fraude: transacciones no reconocidas en su tarjeta de crédito"),
    ("me quedé sin poder entrar a la banca en línea", "El usuario de banca en línea fue bloqueado y no puede entrar"),
    ("quiero un crédito para mi negocio", "Solicitud para préstamo de pequeña empresa para capital de trabajo"),
    ("el cajero se tragó mi tarjeta", "El cajero automático retuvo la tarjeta del cliente"),
    ("necesito mis estados de cuenta anteriores", "Solicitud de copia de estado de cuenta de los últimos 6 meses"),
]


def crear_coleccion(client, nombre):
    try:
        client.delete_collection(nombre)
    except Exception:
        pass
    return client.create_collection(
        nombre,
        configuration={
            "hnsw": {
                "space": "cosine",
                "max_neighbors": M,
                "ef_construction": EF_CONSTRUCTION,
                "ef_search": EF_SEARCH,
            }
        },
    )


def ejecutar():
    print("Cargando modelo de embeddings...")
    modelo = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    embedding_docs = modelo.encode(documentos).astype("float32")
    print("Forma de los embeddings:", embedding_docs.shape)

    client = chromadb.HttpClient(host="localhost", port=PUERTO)
    print(f"Base: Chroma ({client.get_version()}) | HNSW M={M}, ef_construction={EF_CONSTRUCTION}, ef_search={EF_SEARCH}")

    # ------------------------------------------- 1. carga de las 20 frases
    col = crear_coleccion(client, "tickets_banco")
    t0 = time.perf_counter()
    col.add(
        ids=[str(i) for i in range(len(documentos))],
        embeddings=embedding_docs.tolist(),
        documents=documentos,
        metadatas=[{"categoria": c} for c in categorias],
    )
    t_carga = time.perf_counter() - t0
    print(f"\n[1] Carga de {col.count()} frases: {t_carga:.3f} s")

    def buscar(consulta_vec, k, where=None):
        res = col.query(query_embeddings=[consulta_vec], n_results=k, where=where)
        return res["documents"][0], [1 - d for d in res["distances"][0]]

    # ------------------------------------------- 2. aciertos y latencia
    vectores_q = [modelo.encode(p).astype("float32").tolist() for p, _ in preguntas_prueba]
    buscar(vectores_q[0], 3)  # calentamiento

    print("\n[2] Aciertos y tiempos (solo la consulta a Chroma)")
    resumen = {}
    for k in (3, 5, 10):
        aciertos, tiempos = 0, []
        for (pregunta, esperado), v in zip(preguntas_prueba, vectores_q):
            textos, _ = buscar(v, k)
            if esperado in textos:
                aciertos += 1
            for _ in range(REPETICIONES):
                t0 = time.perf_counter()
                buscar(v, k)
                tiempos.append((time.perf_counter() - t0) * 1000)
        resumen[k] = (aciertos, min(tiempos), max(tiempos), statistics.mean(tiempos))
        print(f"  K={k:<2} aciertos {aciertos}/{len(preguntas_prueba)} | "
              f"min {min(tiempos):.1f} | max {max(tiempos):.1f} | "
              f"prom {statistics.mean(tiempos):.1f} ms")

    print("\n  Detalle (K=3):")
    for (pregunta, esperado), v in zip(preguntas_prueba, vectores_q):
        textos, scores = buscar(v, 3)
        print(("  OK  " if esperado in textos else "  FALLO"), pregunta)
        if esperado not in textos:
            print("     Esperado:", esperado)
            for t, s in zip(textos, scores):
                print(f"     {s:.3f} | {t}")

    # --------------------------------------------- 3. filtro por categoria
    print("\n[3] Filtro por metadatos")
    v = modelo.encode("alguien uso mi tarjeta sin permiso").astype("float32").tolist()
    print("  SIN filtro:")
    for t, s in zip(*buscar(v, 3)):
        print(f"    {s:.3f} | {t}")
    print("  CON filtro categoria=fraude:")
    for t, s in zip(*buscar(v, 3, where={"categoria": "fraude"})):
        print(f"    {s:.3f} | {t}")

    # ------------------------------------- 4. escalabilidad (20000 vectores)
    bench = crear_coleccion(client, "bench")
    rng = np.random.default_rng(SEMILLA)
    datos = rng.normal(size=(N_BENCH, 384)).astype("float32")

    t0 = time.perf_counter()
    for i in range(0, N_BENCH, 1000):
        fin = min(i + 1000, N_BENCH)
        bench.add(ids=[str(j) for j in range(i, fin)], embeddings=datos[i:fin].tolist())
    t_bench = time.perf_counter() - t0
    print(f"\n[4] Carga de {N_BENCH} vectores: {t_bench:.2f} s")

    consulta = rng.normal(size=384).astype("float32").tolist()
    tiempos = []
    for _ in range(100):
        t0 = time.perf_counter()
        bench.query(query_embeddings=[consulta], n_results=10)
        tiempos.append((time.perf_counter() - t0) * 1000)
    print(f"    Latencia con {N_BENCH} vectores: promedio {np.mean(tiempos):.1f} ms | "
          f"p95 {np.percentile(tiempos, 95):.1f} ms")

    # ------------------------- 5. recall@10: aproximada (HNSW) vs exacta
    norm = datos / np.linalg.norm(datos, axis=1, keepdims=True)
    aciertos, total = 0, 0
    for _ in range(50):
        q = rng.normal(size=384).astype("float32")
        aprox = {int(i) for i in bench.query(query_embeddings=[q.tolist()], n_results=10)["ids"][0]}
        sims = norm @ (q / np.linalg.norm(q))
        exacta = set(np.argsort(-sims)[:10].tolist())
        aciertos += len(aprox & exacta)
        total += 10
    recall = aciertos / total
    print(f"[5] Recall@10 (HNSW vs exacta): {recall:.2%}")

    # --------------------------------------------------------------- resumen
    print("\n================ RESUMEN CHROMA ================")
    print(f"Carga 20 frases:        {t_carga:.3f} s")
    for k, (a, mn, mx, pr) in resumen.items():
        print(f"K={k:<2} aciertos {a}/5 | tiempo min/max/prom: {mn:.1f} / {mx:.1f} / {pr:.1f} ms")
    print(f"Carga {N_BENCH} vectores: {t_bench:.2f} s")
    print(f"Latencia {N_BENCH} (prom): {np.mean(tiempos):.1f} ms | p95 {np.percentile(tiempos, 95):.1f} ms")
    print(f"Recall@10:              {recall:.2%}")


if __name__ == "__main__":
    buf = io.StringIO()
    with redirect_stdout(buf):
        ejecutar()
    texto = buf.getvalue()
    print(texto, end="")
    SALIDA_TXT.write_text(texto, encoding="utf-8")
    print(f"\nGuardado en {SALIDA_TXT}")
