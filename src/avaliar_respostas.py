import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from busca import Buscador  
from llm import PAUSA_ENTRE_CHAMADAS, criar_cliente, ler_cache, salvar_no_cache 
from responder import responder 

PERGUNTAS_TESTE = Path("data/perguntas_avaliacao.jsonl")
SAIDA = Path("data/avaliacao_respostas.jsonl")
RESPONDIVEIS = 15  
SEMENTE = 42

# Perguntas que nenhuma tese de Computação responde: o sistema deve dizer que não sabe
FORA_DO_ESCOPO = [
    "Quem ganhou a Copa do Mundo de futebol de 2002?",
    "Qual é a capital da Austrália?",
    "Como fazer pão de queijo mineiro?",
    "Qual filme ganhou o Oscar de melhor filme em 2020?",
    "Quanto custa uma passagem de ônibus em São Paulo?",
]


def montar_perguntas() -> list[dict]:
    with open(PERGUNTAS_TESTE, encoding="utf-8") as arquivo:
        teste = pd.DataFrame([json.loads(linha) for linha in arquivo]).drop_duplicates("id", keep="last")
    sorteadas = teste.sample(n=RESPONDIVEIS, random_state=SEMENTE)
    perguntas = [{"pergunta": r.pergunta, "tipo": "respondivel", "id_alvo": int(r.id)}
                 for r in sorteadas.itertuples()]
    perguntas += [{"pergunta": p, "tipo": "fora_do_escopo", "id_alvo": None} for p in FORA_DO_ESCOPO]
    return perguntas


def main() -> int:
    perguntas = montar_perguntas()
    feitas = {registro["pergunta"] for registro in ler_cache(SAIDA)}
    pendentes = [p for p in perguntas if p["pergunta"] not in feitas]
    print(f"{len(perguntas)} perguntas; {len(feitas)} já respondidas, {len(pendentes)} pendentes")
    if not pendentes:
        return 0

    buscador = Buscador()
    cliente = criar_cliente()
    for numero, item in enumerate(pendentes, start=1):
        resultado = responder(item["pergunta"], buscador, cliente, registrar=False)
        salvar_no_cache(SAIDA, [{**item, **resultado}])
        situacao = "respondeu" if resultado["suficiente"] else "disse que não sabe"
        print(f"{numero}/{len(pendentes)} ({item['tipo']}): {situacao}")
        time.sleep(PAUSA_ENTRE_CHAMADAS)

    print(f"Respostas em {SAIDA}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
