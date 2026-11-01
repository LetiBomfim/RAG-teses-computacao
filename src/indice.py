"""Monta o índice de busca semântica: um vetor por resumo, pronto para ser comparado com a pergunta.

Um embedding é um vetor de números que representa o sentido de um texto: textos sobre o mesmo
assunto ficam com vetores próximos. Para buscar, a pergunta também vira um vetor, e os resumos
mais próximos dela são os mais relevantes.

Os vetores dos resumos já foram calculados no projeto de evolução temática, este script só os prepara para a busca: confere, 
reduz o tamanho pela metade e salva junto com o id de cada trabalho. Se o arquivo não existir, ele é gerado aqui, 
com o mesmo modelo.

Uso (a partir da raiz do repositório):
    python src/indice.py
    python src/indice.py --testar "redes neurais para diagnóstico de câncer"
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("data/teses_computacao_2010_2024.csv.gz")
EMBEDDINGS = Path("data/embeddings.npz")  
INDICE = Path("data/indice.npz")
MODELO = "paraphrase-multilingual-MiniLM-L12-v2"


def limpar(texto: str) -> str:
    """A mesma limpeza usada antes dos embeddings: minúsculas, sem links e espaços extras."""
    texto = re.sub(r"https?://\S+|www\.\S+", " ", str(texto).lower())
    return re.sub(r"\s+", " ", texto).strip()


def gerar_embeddings(base: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Só roda se o arquivo do outro projeto não estiver em data/. Demora de 15 a 40 minutos."""
    from sentence_transformers import SentenceTransformer

    com_resumo = base[base["resumo"].notna() & (base["resumo"].str.len() > 100)]
    print(f"Gerando embeddings de {len(com_resumo)} resumos com {MODELO} (demora)...")
    modelo = SentenceTransformer(MODELO)
    vetores = modelo.encode(com_resumo["resumo"].map(limpar).tolist(), batch_size=16,
                            show_progress_bar=True, normalize_embeddings=True)
    return com_resumo.index.to_numpy(), vetores


def testar(pergunta: str, ids: np.ndarray, vetores: np.ndarray, base: pd.DataFrame) -> None:
    """Primeira busca: os 5 resumos mais próximos da pergunta."""
    from sentence_transformers import SentenceTransformer

    modelo = SentenceTransformer(MODELO)
    vetor_pergunta = modelo.encode(limpar(pergunta), normalize_embeddings=True)
    # Com vetores de tamanho 1, o produto escalar é a similaridade de cosseno:
    # 1 = mesmo sentido, 0 = nada a ver
    similaridades = vetores.astype(np.float32) @ vetor_pergunta
    melhores = np.argsort(-similaridades)[:5]
    print(f'\nPergunta: "{pergunta}"\n')
    for posicao in melhores:
        trabalho = base.loc[ids[posicao]]
        print(f"  {similaridades[posicao]:.2f}  {trabalho['titulo'][:90]} "
              f"({trabalho['ies_sigla']}, {trabalho['ano']})")


def main() -> int:
    parser = argparse.ArgumentParser(description="Monta o índice de busca semântica.")
    parser.add_argument("--testar", metavar="PERGUNTA", help="faz uma busca de teste no índice")
    args = parser.parse_args()

    if not BASE.exists():
        print(f"Arquivo não encontrado: {BASE}. Copie do projeto de evolução temática (veja o README).")
        return 1
    base = pd.read_csv(BASE, usecols=["titulo", "resumo", "ies_sigla", "ano"])

    if EMBEDDINGS.exists():
        arquivo = np.load(EMBEDDINGS)
        ids, vetores = arquivo["ids"], arquivo["vetores"]
        print(f"Embeddings lidos de {EMBEDDINGS}: {vetores.shape[0]} resumos, {vetores.shape[1]} dimensões")
    else:
        ids, vetores = gerar_embeddings(base)

    assert ids.max() < len(base), "Há ids fora da base: os embeddings são de outra versão dos dados?"
    normas = np.linalg.norm(vetores, axis=1)
    assert np.allclose(normas, 1, atol=1e-3), "Os vetores deveriam estar normalizados (tamanho 1)"

    INDICE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(INDICE, ids=ids, vetores=vetores.astype(np.float16))
    tamanho_mb = INDICE.stat().st_size / 1e6
    print(f"Índice salvo em {INDICE} ({tamanho_mb:.0f} MB)")

    if args.testar:
        testar(args.testar, ids, vetores, base)
    return 0


if __name__ == "__main__":
    sys.exit(main())
