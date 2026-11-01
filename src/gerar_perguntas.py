import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel

sys.path.append(str(Path(__file__).resolve().parent))
from indice import BASE, INDICE  # noqa: E402
from llm import PAUSA_ENTRE_CHAMADAS, criar_cliente, gerar_json, ler_cache, salvar_no_cache  # noqa: E402

SAIDA = Path("data/perguntas_avaliacao.jsonl")
POR_ANO = 4        # 4 trabalhos por ano, 2010–2024: 60 perguntas
POR_CHAMADA = 10
SEMENTE = 42
LIMITE_RESUMO = 1500  # caracteres; o começo do resumo basta para entender o tema


class Pergunta(BaseModel):
    id: int
    pergunta: str


class Lote(BaseModel):
    perguntas: list[Pergunta]


INSTRUCOES = """Você vai receber títulos e resumos de teses e dissertações brasileiras de \
Computação, cada um com um id.

Para cada trabalho, escreva UMA pergunta em português que uma pessoa faria a um buscador de \
teses e que esse trabalho responde. Regras:
- Escreva como alguém que procura trabalhos sobre o assunto, sem conhecer este trabalho.
  Exemplo de formato: "Que pesquisas usam aprendizado de máquina para prever evasão escolar?"
- Não copie o título. Use suas próprias palavras e, quando possível, sinônimos no lugar dos
  termos mais característicos do título.
- Fale do problema e da abordagem, sem citar nomes de ferramentas criadas no trabalho, autor,
  instituição ou ano.
- Entre 8 e 25 palavras.

Responda com uma pergunta para cada id recebido.

"""


def montar_amostra() -> pd.DataFrame:
    """Sorteia os trabalhos entre os que estão no índice, de forma reprodutível."""
    ids = np.load(INDICE)["ids"]
    base = pd.read_csv(BASE, usecols=["ano", "titulo", "resumo"])
    no_indice = base.loc[ids].reset_index(names="id")
    return no_indice.groupby("ano").sample(n=POR_ANO, random_state=SEMENTE)


def main() -> int:
    amostra = montar_amostra()
    feitos = {registro["id"] for registro in ler_cache(SAIDA)}
    pendentes = amostra[~amostra["id"].isin(feitos)]
    print(f"Amostra de {len(amostra)} trabalhos; {len(feitos)} já com pergunta, "
          f"{len(pendentes)} pendentes")
    if pendentes.empty:
        return 0

    cliente = criar_cliente()
    for inicio in range(0, len(pendentes), POR_CHAMADA):
        lote = pendentes.iloc[inicio:inicio + POR_CHAMADA]
        prompt = INSTRUCOES + "\n\n".join(
            f"[id {r.id}]\nTítulo: {r.titulo}\nResumo: {str(r.resumo)[:LIMITE_RESUMO]}"
            for r in lote.itertuples())
        resposta = gerar_json(cliente, prompt, Lote)

        titulos = dict(zip(lote["id"], lote["titulo"]))
        validos = [{**p.model_dump(), "titulo": titulos[p.id]}
                   for p in resposta.perguntas if p.id in titulos]
        salvar_no_cache(SAIDA, validos)
        print(f"Lote {inicio // POR_CHAMADA + 1}: {len(validos)}/{len(lote)} perguntas")
        time.sleep(PAUSA_ENTRE_CHAMADAS)

    respondidos = {registro["id"] for registro in ler_cache(SAIDA)}
    faltando = len(set(amostra["id"]) - respondidos)
    if faltando:
        print(f"{faltando} trabalhos ficaram sem pergunta; rode de novo para completar.")
    print(f"Perguntas em {SAIDA}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
