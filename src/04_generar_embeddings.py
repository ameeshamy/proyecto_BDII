import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

ENTRADA = Path("data/chunks/chunks_limpios.json")
SALIDA = Path("data/embeddings.npy")

chunks = json.loads(ENTRADA.read_text(encoding="utf-8"))
textos = [c["contenido"] for c in chunks]

modelo = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
embeddings = modelo.encode(textos, show_progress_bar=True)

np.save(SALIDA, embeddings)
print("Forma:", embeddings.shape)
print("Dimensiones:", embeddings.shape[1])
print("Ejemplo (primeros 5 valores):", embeddings[0][:5])