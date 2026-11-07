import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, ValidationError

load_dotenv()

MODELO = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
PAUSA_ENTRE_CHAMADAS = 6
TENTATIVAS = 5
LIMITE_RESPOSTA_MS = 6 * 60 * 1000
FALHAS_SERVIDOR = 2  # 503 ou timeout seguidos antes de desistir 

class CotaEsgotada(RuntimeError):
    """A cota diária acabou: não adianta tentar de novo hoje."""


class ServidorIndisponivel(RuntimeError):
    """O Gemini está sobrecarregado ou não responde: melhor tentar mais tarde."""


def criar_cliente() -> genai.Client:
    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("Defina GEMINI_API_KEY no arquivo .env (veja o README).")
    return genai.Client(http_options=types.HttpOptions(timeout=LIMITE_RESPOSTA_MS))


def gerar_json(cliente, prompt: str, esquema: type[BaseModel]) -> BaseModel:
    """Envia o prompt e devolve a resposta já validada pelo esquema Pydantic.

    O esquema é enviado ao modelo para ele responder exatamente nesse formato. Mesmo assim a
    resposta é validada aqui, e a chamada é repetida se vier fora do formato ou se a API recusar
    por excesso de pedidos (erro 429), esperando cada vez mais entre as tentativas. Se a cota do
    dia acabar, ou se o servidor estiver sobrecarregado (503) duas vezes seguidas, desiste logo.
    """
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_json_schema=esquema.model_json_schema(),
        temperature=0,  # respostas o mais estáveis possível para a mesma entrada
    )
    falhas_servidor = 0
    for tentativa in range(1, TENTATIVAS + 1):
        try:
            resposta = cliente.models.generate_content(model=MODELO, contents=prompt, config=config)
            return esquema.model_validate_json(resposta.text)
        except ValidationError as erro:
            motivo = f"resposta fora do formato ({erro.error_count()} erros)"
        except Exception as erro:  # erros de rede ou da API
            motivo = f"{type(erro).__name__}: {str(erro)[:120]}"
            codigo = getattr(erro, "code", None)
            texto = str(erro)

            # Cota do dia esgotada: insistir não adianta e só ocupa o terminal até amanhã
            if codigo == 429 and "PerDay" in texto:
                raise CotaEsgotada("A cota diária do Gemini acabou. Rode de novo amanhã; "
                                   "o que já foi salvo não se perde.") from erro

            # Servidor sobrecarregado (503) ou sem resposta a tempo: cada tentativa pode contar na
            # cota mesmo sem resposta, então desistimos logo e deixamos para rodar mais tarde
            if (codigo is not None and codigo >= 500) or "Timeout" in type(erro).__name__:
                falhas_servidor += 1
                if falhas_servidor >= FALHAS_SERVIDOR:
                    raise ServidorIndisponivel(
                        f"O Gemini não respondeu em {falhas_servidor} tentativas ({motivo}). "
                        "Tente de novo mais tarde; o que já foi salvo não se perde.") from erro
        espera = PAUSA_ENTRE_CHAMADAS * 2 ** tentativa
        print(f"  tentativa {tentativa} falhou ({motivo}); nova tentativa em {espera}s")
        time.sleep(espera)
    raise RuntimeError(f"Falha após {TENTATIVAS} tentativas. Último erro: {motivo}")


def ler_cache(caminho: Path) -> list[dict]:
    """Respostas já recebidas, uma por linha. Permite interromper e continuar sem pagar de novo."""
    if not caminho.exists():
        return []
    with open(caminho, encoding="utf-8") as arquivo:
        return [json.loads(linha) for linha in arquivo if linha.strip()]


def salvar_no_cache(caminho: Path, registros: list[dict]) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with open(caminho, "a", encoding="utf-8") as arquivo:
        for registro in registros:
            arquivo.write(json.dumps(registro, ensure_ascii=False) + "\n")