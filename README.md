# Busca com IA nas teses de Computação do Brasil (RAG)

Faça uma pergunta em português sobre a pesquisa brasileira em Computação e receba uma resposta
escrita a partir das teses e dissertações reais, **com a citação de cada trabalho usado**. Se os
documentos encontrados não respondem à pergunta, o sistema diz isso.

A base são os resumos de 23 mil teses e dissertações de Computação (2010–2024) do **Catálogo de
Teses e Dissertações da CAPES**, já tratados no projeto
[Evolução temática da pós-graduação em Computação](https://github.com/LetiBomfim/TAC_Pesquisa_temas_computacao).

## Backlog/Roadmap

- [X] Estrutura do repositório
- [ ] Índice de busca: embeddings e BM25
- [ ] Busca híbrida e avaliação da recuperação
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

