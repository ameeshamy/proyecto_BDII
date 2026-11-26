import chromadb
from sentence_transformers import SentenceTransformer

modelo = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
client = chromadb.HttpClient(host="localhost", port=8001)
col = client.get_collection("leyes")

pregunta = "¿Cuántos días de vacaciones le corresponden a un trabajador?"
q = modelo.encode(pregunta).tolist()
res = col.query(query_embeddings=[q], n_results=5)

for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
    print(f"Similitud: {1 - dist:.3f} | {meta['documento']} art. {meta['articulo']}")
    print(f"   {doc[:200]}\n")