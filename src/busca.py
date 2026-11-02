import argparse
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

sys.path.append(str(Path(__file__).resolve().parent))
from indice import BASE, INDICE, MODELO, limpar  # noqa: E402

MODOS = ("semantica", "bm25", "hibrida")
RRF_K = 60  # valor padrão da literatura; diminui o peso da diferença entre as primeiras posições
CANDIDATOS = 100  # quantos resultados de cada busca entram na fusão

# Palavras muito comuns, que não ajudam a distinguir um resumo de outro
STOPWORDS = set("""
a ao aos as com como da das de do dos e em entre era essa esse esta este eu foi ha isso isto
ja mais mas mesmo muito na nao nas no nos o os ou para pela pelas pelo pelos por qual quais
que se sem ser seu sua sao tambem tem um uma umas uns sobre ate apos quando onde cada outro
outros outra outras ser sido sendo foram pode podem ainda assim bem the of and to in for on
with by an is are this that from be as at it its or which
trabalho trabalhos estudo pesquisa dissertacao tese presente proposta propoe resultados
""".split())


def tokenizar(texto: str) -> list[str]:
    """Minúsculas, sem acentos, só palavras com 2 letras ou mais, sem stopwords.

    Tirar os acentos faz "diagnóstico" e "diagnostico" contarem como a mesma palavra.
    """
    texto = unicodedata.normalize("NFKD", str(texto).lower())
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    return [t for t in re.findall(r"[a-z0-9]{2,}", texto) if t not in STOPWORDS]


class Buscador:
    """Carrega o índice uma vez e responde a quantas buscas forem necessárias."""

    def __init__(self, modelo=None):
        arquivo = np.load(INDICE)
        self.ids = arquivo["ids"]
        self.vetores = arquivo["vetores"].astype(np.float32)

        colunas = ["ano", "grau", "regiao", "ies_sigla", "programa", "titulo", "palavras_chave", "resumo"]
        base = pd.read_csv(BASE, usecols=colunas)
        self.documentos = base.loc[self.ids].reset_index(names="id")

        # No BM25 o título e as palavras-chave entram junto com o resumo: é neles que costuma
        # estar o domínio do trabalho ("... para detecção de câncer de mama")
        textos = (self.documentos["titulo"].fillna("") + " " +
                  self.documentos["palavras_chave"].fillna("") + " " +
                  self.documentos["resumo"].fillna(""))
        self.bm25 = BM25Okapi([tokenizar(t) for t in textos])

        self._modelo = modelo

    @property
    def modelo(self):
        # O modelo só é carregado na primeira busca semântica. Assim, quem usa só o BM25 (como o
        # app publicado) não precisa instalar o torch nem esperar o modelo carregar.
        if self._modelo is None:
            from sentence_transformers import SentenceTransformer
            self._modelo = SentenceTransformer(MODELO)
        return self._modelo

    def _semantica(self, pergunta: str) -> np.ndarray:
        vetor = self.modelo.encode(limpar(pergunta), normalize_embeddings=True)
        return self.vetores @ vetor

    def _bm25(self, pergunta: str) -> np.ndarray:
        return self.bm25.get_scores(tokenizar(pergunta))

    def ranking(self, pergunta: str, modo: str = "hibrida", k: int = 10) -> list[int]:
        """Posições (no índice) dos k documentos mais relevantes, do melhor para o pior."""
        if modo == "semantica":
            return list(np.argsort(-self._semantica(pergunta))[:k])
        if modo == "bm25":
            return list(np.argsort(-self._bm25(pergunta))[:k])
        if modo != "hibrida":
            raise ValueError(f"modo deve ser um destes: {MODOS}")

        pontos: dict[int, float] = {}
        for notas in (self._semantica(pergunta), self._bm25(pergunta)):
            for posicao, documento in enumerate(np.argsort(-notas)[:CANDIDATOS], start=1):
                pontos[documento] = pontos.get(documento, 0.0) + 1 / (RRF_K + posicao)
        return sorted(pontos, key=pontos.get, reverse=True)[:k]

    def buscar(self, pergunta: str, modo: str = "hibrida", k: int = 10) -> pd.DataFrame:
        """Os k documentos mais relevantes, com os dados de cada trabalho."""
        return self.documentos.iloc[self.ranking(pergunta, modo, k)]


def main() -> int:
    parser = argparse.ArgumentParser(description="Busca nos resumos das teses.")
    parser.add_argument("pergunta")
    parser.add_argument("--modo", choices=MODOS, default="hibrida")
    parser.add_argument("-k", type=int, default=5, help="quantos resultados mostrar")
    args = parser.parse_args()

    print("Carregando o índice...")
    resultados = Buscador().buscar(args.pergunta, args.modo, args.k)
    print(f'\nPergunta: "{args.pergunta}" (busca {args.modo})\n')
    for posicao, trabalho in enumerate(resultados.itertuples(), start=1):
        print(f"  {posicao}. {trabalho.titulo[:95]} ({trabalho.ies_sigla}, {trabalho.ano})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
