import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

sys.path.append(str(Path(__file__).resolve().parent))
from busca import MODOS, Buscador  # noqa: E402
from llm import criar_cliente, gerar_json, salvar_no_cache  # noqa: E402

REGISTRO = Path("data/respostas.jsonl")  
DOCUMENTOS_POR_RESPOSTA = 5
MODO_PADRAO = "bm25"  
LIMITE_RESUMO = 3000  # caracteres por resumo; com 1.200, a avaliação mostrou recusas por resumos cortados 


class Resposta(BaseModel):
    suficiente: bool = Field(description="true se os documentos permitem responder à pergunta")
    resposta: str = Field(description="a resposta, com o número do documento entre colchetes após cada afirmação")
    citacoes: list[int] = Field(description="os números dos documentos usados na resposta")


INSTRUCOES = """Você é um assistente que responde perguntas sobre a pesquisa brasileira em \
Computação, usando SOMENTE os resumos de teses e dissertações fornecidos abaixo.

Regras:
- Use apenas as informações dos documentos. Não use conhecimento próprio e não invente trabalhos.
- Depois de cada afirmação, indique o documento de onde ela veio com o número entre colchetes,
  por exemplo [2]. Uma afirmação apoiada em dois documentos leva os dois: [1][3].
- Se nenhum documento ajuda a responder, marque "suficiente" como false e diga, em uma frase, que
  os trabalhos encontrados não tratam do assunto. Se só parte da pergunta é respondida, responda
  essa parte e diga o que ficou sem resposta.
- Responda em português, em 3 a 6 frases, de forma direta. Pode citar o título dos trabalhos.
- Em "citacoes", liste os números de todos os documentos citados na resposta.

"""


def montar_prompt(pergunta: str, documentos) -> str:
    blocos = [
        f"[{numero}] {doc.titulo} ({doc.grau}, {doc.ies_sigla}, {doc.ano})\n"
        f"{str(doc.resumo)[:LIMITE_RESUMO]}"
        for numero, doc in enumerate(documentos.itertuples(), start=1)
    ]
    return INSTRUCOES + "Documentos:\n\n" + "\n\n".join(blocos) + f"\n\nPergunta: {pergunta}"


def verificar_citacoes(resposta: Resposta, total: int) -> dict:
    """Confere se as citações apontam para documentos que existem.

    O LLM pode citar um [7] quando só recebeu 5 documentos, ou citar no texto um número que não
    pôs na lista. Nada disso é aceito em silêncio: as citações inválidas são retiradas do texto e
    contadas, para a avaliação.
    """
    no_texto = {int(n) for n in re.findall(r"\[(\d+)\]", resposta.resposta)}
    citadas = no_texto | set(resposta.citacoes)
    validas = {n for n in citadas if 1 <= n <= total}
    invalidas = citadas - validas

    texto = resposta.resposta
    for numero in invalidas:
        texto = re.sub(rf"\s*\[{numero}\]", "", texto)
    return {"texto": texto.strip(), "citacoes": sorted(validas), "citacoes_invalidas": sorted(invalidas),
            "sem_citacao": resposta.suficiente and not validas}


def responder(pergunta: str, buscador: Buscador, cliente, modo: str = MODO_PADRAO,
              k: int = DOCUMENTOS_POR_RESPOSTA, registrar: bool = True) -> dict:
    """Busca os documentos, pede a resposta ao Gemini e confere as citações."""
    documentos = buscador.buscar(pergunta, modo, k)
    resposta = gerar_json(cliente, montar_prompt(pergunta, documentos), Resposta)
    conferida = verificar_citacoes(resposta, len(documentos))

    resultado = {
        "pergunta": pergunta,
        "modo": modo,
        "suficiente": resposta.suficiente,
        "resposta": conferida["texto"],
        "citacoes": conferida["citacoes"],
        "citacoes_invalidas": conferida["citacoes_invalidas"],
        "sem_citacao": conferida["sem_citacao"],
        "documentos": documentos["id"].tolist(),  # na ordem em que foram numerados: [1], [2]...
        "data": datetime.now().isoformat(timespec="seconds"),
    }
    if registrar:
        salvar_no_cache(REGISTRO, [resultado])
    return resultado


def main() -> int:
    parser = argparse.ArgumentParser(description="Responde a uma pergunta citando as teses.")
    parser.add_argument("pergunta")
    parser.add_argument("--modo", choices=MODOS, default=MODO_PADRAO)
    args = parser.parse_args()

    print("Carregando o índice...")
    buscador = Buscador()
    resultado = responder(args.pergunta, buscador, criar_cliente(), args.modo)

    print(f"\nPergunta: {resultado['pergunta']}\n")
    print(resultado["resposta"])
    if resultado["citacoes_invalidas"]:
        print(f"\n(Citações inválidas retiradas: {resultado['citacoes_invalidas']})")

    print("\nDocumentos usados:")
    documentos = buscador.documentos.set_index("id").loc[resultado["documentos"]]
    for numero, (id_, doc) in enumerate(documentos.iterrows(), start=1):
        marca = "*" if numero in resultado["citacoes"] else " "
        print(f" {marca}[{numero}] {doc['titulo'][:90]} ({doc['ies_sigla']}, {doc['ano']})")
    print("\n* = citado na resposta")
    return 0


if __name__ == "__main__":
    sys.exit(main())
