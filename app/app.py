import os
import re
import sys
from pathlib import Path

import streamlit as st

RAIZ = Path(__file__).resolve().parent.parent
os.chdir(RAIZ)  # os scripts usam caminhos a partir da raiz do repositório
sys.path.append(str(RAIZ / "src"))

import llm  
from busca import Buscador  
from responder import responder  

PERGUNTAS_POR_SESSAO = 5  
EXEMPLOS = {  # texto do botão: pergunta
    "Evasão escolar": "Que pesquisas usam aprendizado de máquina para prever evasão escolar?",
    "Acessibilidade para cegos": "Quais trabalhos estudam acessibilidade para pessoas cegas?",
    "IA na agricultura": "Como a visão computacional tem sido usada na agricultura?",
}

llm.TENTATIVAS = 2
llm.PAUSA_ENTRE_CHAMADAS = 3

st.set_page_config(page_title="Busca com IA nas teses de Computação", page_icon="🔎", layout="centered")


def configurar_chave() -> bool:
    """No Streamlit Cloud a chave vem dos Secrets; no computador, do arquivo .env."""
    try:
        if "GEMINI_API_KEY" in st.secrets:
            os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
    except FileNotFoundError:
        pass  
    except Exception:  
        pass
    return bool(os.getenv("GEMINI_API_KEY"))


@st.cache_resource(show_spinner="Carregando as 23 mil teses (só na primeira vez)...")
def carregar_buscador() -> Buscador:
    return Buscador()  


@st.cache_resource
def carregar_cliente():
    return llm.criar_cliente()


@st.cache_data(show_spinner=False, max_entries=200)
def responder_com_cache(pergunta: str) -> dict:
    # A mesma pergunta feita duas vezes não gasta outra chamada
    return responder(pergunta, carregar_buscador(), carregar_cliente(), registrar=False)


def legivel(texto: str) -> str:
    """Muitos títulos e resumos da CAPES estão em maiúsculas; aqui viram frases normais."""
    texto = " ".join(str(texto).split())
    if not texto.isupper():
        return texto
    frases = re.split(r"(?<=[.!?])\s+", texto.lower())
    return " ".join(frase[:1].upper() + frase[1:] for frase in frases)


def mostrar_documentos(ids: list[int], citados: list[int]) -> None:
    documentos = carregar_buscador().documentos.set_index("id").loc[ids]
    for numero, (_, doc) in enumerate(documentos.iterrows(), start=1):
        marca = "📌 " if numero in citados else ""
        with st.expander(f"{marca}[{numero}] {legivel(doc['titulo'])}"):
            st.caption(f"{doc['grau']} · {doc['ies_sigla']} · {doc['ano']} · {doc['programa']}")
            st.write(legivel(doc["resumo"]))


# --------------------------------------------------------------------------------------------
# Página
# --------------------------------------------------------------------------------------------

st.title("🔎 Busca com IA nas teses de Computação")
st.markdown(
    "Pergunte sobre a pesquisa brasileira em Computação. A resposta é escrita a partir de "
    "**23 mil teses e dissertações** do Catálogo da CAPES (2010–2024), com o número de cada "
    "trabalho usado entre colchetes. Se os trabalhos encontrados não respondem, o sistema avisa "
    "em vez de inventar."
)
st.caption("[Código, metodologia e avaliação no GitHub](https://github.com/LetiBomfim/RAG-teses-computacao)")

if "pergunta" not in st.session_state:
    st.session_state.pergunta = ""
if "usadas" not in st.session_state:
    st.session_state.usadas = 0

st.write("Exemplos:")
clicou_exemplo = False
colunas = st.columns(len(EXEMPLOS))
for coluna, (rotulo, exemplo) in zip(colunas, EXEMPLOS.items()):
    if coluna.button(rotulo, width="stretch"):
        st.session_state.pergunta = exemplo
        clicou_exemplo = True

with st.form("pergunta_form"):
    pergunta = st.text_input("Sua pergunta", key="pergunta",
                             placeholder="ex.: Quais trabalhos usam IA para detectar fake news?")
    enviar = st.form_submit_button("Perguntar", type="primary")

if (enviar or clicou_exemplo) and pergunta.strip():
    buscador = carregar_buscador()
    if not configurar_chave():
        st.error("A chave do Gemini não está configurada. Veja o README do projeto.")
        st.stop()
    if st.session_state.usadas >= PERGUNTAS_POR_SESSAO:
        st.warning(f"Limite de {PERGUNTAS_POR_SESSAO} perguntas por visita, para não esgotar a cota "
                   "gratuita do Gemini. Os trabalhos encontrados aparecem abaixo.")
        mostrar_documentos(buscador.buscar(pergunta, "bm25", 5)["id"].tolist(), [])
        st.stop()

    with st.spinner("Buscando as teses e escrevendo a resposta..."):
        try:
            resultado = responder_com_cache(pergunta.strip())
            st.session_state.usadas += 1
        except Exception:
            resultado = None

    if resultado is None:
        st.warning("O Gemini não respondeu agora (a cota gratuita pode ter acabado). "
                   "Veja os trabalhos que a busca encontrou:")
        mostrar_documentos(buscador.buscar(pergunta, "bm25", 5)["id"].tolist(), [])
    else:
        st.subheader("Resposta")
        if resultado["suficiente"]:
            st.markdown(resultado["resposta"])
        else:
            st.info(resultado["resposta"])
        st.subheader("Trabalhos encontrados")
        st.caption("📌 = citado na resposta. Clique para ler o resumo.")
        mostrar_documentos(resultado["documentos"], resultado["citacoes"])
