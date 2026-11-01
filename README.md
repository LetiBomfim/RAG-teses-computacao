# Busca com IA nas teses de Computação do Brasil (RAG)

Faça uma pergunta em português sobre a pesquisa brasileira em Computação e receba uma resposta
escrita a partir das teses e dissertações reais, **com a citação de cada trabalho usado**. Se os
documentos encontrados não respondem à pergunta, o sistema diz isso em vez de inventar.

A base são os resumos de 23 mil teses e dissertações de Computação (2010–2024) do **Catálogo de
Teses e Dissertações da CAPES**, já tratados no projeto
[Evolução temática da pós-graduação em Computação](https://github.com/LetiBomfim/TAC_Pesquisa_temas_computacao).

## Backlog/Roadmap

- [X] Estrutura do repositório
- [X] Índice de busca semântica
- [ ] Busca híbrida (semântica + BM25) e avaliação da recuperação
- [ ] Geração de respostas com citações verificadas
- [ ] Avaliação das respostas
- [ ] App de perguntas e respostas

## Como rodar

Requer Python 3.10 ou superior.

```bash
python -m venv .venv
source .venv/bin/activate         # Para Windows é: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env              # e cole sua chave do Gemini no .env
```

A chave do Gemini é gratuita e pode ser criada no
[Google AI Studio](https://aistudio.google.com/apikey). O `.env` não é versionado.

### 1. Trazer os dados

A base de teses (`data/teses_computacao_2010_2024.csv.gz`) já vem no repositório. Os embeddings
dos resumos foram calculados no projeto de
[evolução temática](https://github.com/LetiBomfim/TAC_Pesquisa_temas_computacao) (etapa
`src/embeddings.py`). Se você tem aquele projeto, copie o arquivo para não recalcular:

```bash
cp ../TAC_Pesquisa_temas_computacao/data/processed/embeddings.npz data/
```

### 2. Montar o índice

```bash
python src/indice.py
python src/indice.py --testar "redes neurais para diagnóstico de câncer"
```

O script confere os vetores, reduz o tamanho deles pela metade (de `float32` para `float16`, o
que muda a similaridade só na terceira casa decimal) e salva `data/indice.npz`, com 16 MB, que
vai para o repositório. Sem o arquivo de embeddings, ele os gera com o mesmo modelo
(`paraphrase-multilingual-MiniLM-L12-v2`), o que leva de 15 a 40 minutos em CPU.

Com `--testar`, ele faz uma primeira busca: transforma a pergunta num vetor com o mesmo modelo e
mostra os 5 resumos mais próximos pela similaridade de cosseno.

#