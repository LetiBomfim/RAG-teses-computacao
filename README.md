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
- [X] Busca híbrida (semântica + BM25) e avaliação da recuperação
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
[evolução temática](https://github.com/LetiBomfim/TAC_Pesquisa_temas_computacao). Se você tem
aquele projeto, copie o arquivo para não recalcular:

```bash
cp ../TAC_Pesquisa_temas_computacao/data/processed/embeddings.npz data/
```

### 2. Montar o índice

```bash
python src/indice.py
python src/indice.py --testar "redes neurais para diagnóstico de câncer"
```

### 3. Buscar

```bash
python src/busca.py "redes neurais para diagnóstico de câncer"               
python src/busca.py "redes neurais para diagnóstico de câncer" --modo bm25
python src/busca.py "redes neurais para diagnóstico de câncer" --modo semantica
```

### 4. Avaliar a busca

```bash
python src/gerar_perguntas.py   
```

Depois rode o notebook [`notebooks/01_avaliacao_busca.ipynb`](notebooks/01_avaliacao_busca.ipynb).

## Como funciona

### Índice de busca semântica

[`src/indice.py`](src/indice.py)

Um **embedding** é um vetor de números que representa o sentido de um texto: textos sobre o mesmo
assunto ficam com vetores próximos. Para buscar, a pergunta também vira um vetor, e os resumos mais
próximos dela são os mais relevantes.

Os vetores dos 23.007 resumos já tinham sido calculados no projeto de evolução temática, com o
modelo multilíngue `paraphrase-multilingual-MiniLM-L12-v2`. Calcular de novo levaria de 15 a 40
minutos em CPU, então o script só os prepara para a busca:

- **confere** se cada vetor aponta para um trabalho da base e se todos têm tamanho 1;
- **reduz o tamanho pela metade**, de `float32` para `float16`, o que muda a similaridade só na
  terceira casa decimal. O índice (`data/indice.npz`) fica com 16 MB, pequeno o bastante para ir
  para o GitHub e para o app publicado;
- se o arquivo de embeddings não existir, **gera os vetores** com o mesmo modelo.

A pergunta precisa ser transformada pelo **mesmo modelo** que transformou os resumos, senão os
vetores não são comparáveis. Como todos os vetores têm tamanho 1, a proximidade entre dois deles é
o produto escalar, a **similaridade de cosseno**: perto de 1, mesmo assunto; perto de 0, nada a ver.

### Busca: semântica, BM25 e híbrida

[`src/busca.py`](src/busca.py)

A classe `Buscador` carrega o índice uma vez e busca de três formas:

- **Semântica:** compara o vetor da pergunta com o de cada resumo. Entende sinônimos e paráfrases,
  mas resume o texto num "sentido geral" e pode perder termos específicos. Além disso, o modelo lê
  só o começo de cada resumo (cerca de 100 palavras).
- **BM25:** a busca clássica por palavras, usada por buscadores há décadas. Dá pontos aos
  documentos que contêm as palavras da pergunta, e mais pontos às palavras raras. Considera o
  título, as palavras-chave e o resumo inteiro. Antes, o texto passa a minúsculas, perde os
  acentos ("diagnóstico" e "diagnostico" viram a mesma palavra) e perde as palavras muito comuns
  ("de", "para", "trabalho"). Acha termos exatos, mas não entende sinônimos.
- **Híbrida:** junta as duas listas pela *Reciprocal Rank Fusion* (RRF). Cada documento ganha
  1 / (60 + posição) em cada lista, e os pontos são somados: quem aparece bem nas duas sobe. Como
  usa só as posições, não importa que as notas das duas buscas tenham escalas diferentes.

O primeiro teste mostrou por que vale combinar as duas. Para "redes neurais para diagnóstico de
câncer":

| Busca | O que trouxe nos 5 primeiros |
|---|---|
| Semântica | Só trabalhos sobre redes neurais em geral; nenhum sobre câncer |
| Híbrida | Os 5 sobre câncer: carcinoma de mama, tumores orais, segmentação de núcleos celulares, subclasses de câncer e pólipos |

### Perguntas de teste

[`src/gerar_perguntas.py`](src/gerar_perguntas.py)

Para saber se uma busca é boa, são necessárias perguntas cuja resposta certa já é conhecida. O
script sorteia 60 trabalhos (4 por ano, de forma reprodutível) e pede ao Gemini a pergunta que
alguém faria a um buscador e que cada trabalho responde. Na avaliação, cada busca é testada: o
trabalho que originou a pergunta aparece entre os primeiros resultados?

O LLM é instruído a **não copiar o título** e a usar as próprias palavras. Sem isso, o BM25 levaria
vantagem injusta, porque bastaria repetir os termos do título. Cada chamada leva 10 trabalhos
(6 chamadas no total), e as perguntas ficam salvas em `data/perguntas_avaliacao.jsonl`.

### Cliente do Gemini

[`src/llm.py`](src/llm.py)

As funções comuns para chamar o Gemini, reaproveitadas do projeto de evolução temática:

- **saída em JSON validada:** o esquema da resposta (um modelo Pydantic) é enviado ao Gemini, e a
  resposta é validada de novo ao chegar;
- **`temperature=0`**, para respostas o mais estáveis possível;
- **novas tentativas** com espera crescente quando a API recusa (limite de uso, servidor
  sobrecarregado) ou responde fora do formato, e um **limite de tempo** por chamada;
- **cache em JSONL:** cada resposta é salva assim que chega. Se o script parar no meio, basta
  rodar de novo que ele continua de onde parou, sem repetir chamadas.

## Avaliação da busca

O notebook [`notebooks/01_avaliacao_busca.ipynb`](notebooks/01_avaliacao_busca.ipynb) testa as três
buscas com as 60 perguntas de teste. Para cada pergunta, sabemos qual trabalho a originou, e vemos
em que posição ele aparece.

- **Recall@5:** em quantas perguntas o trabalho certo aparece entre os 5 primeiros resultados. É
  a medida principal, porque são mais ou menos os documentos que o LLM vai ler.
- **MRR** (*mean reciprocal rank*): a média de 1/posição. Vale 1 se o trabalho certo vem sempre em
  1º lugar e 0 se fica fora dos 10 primeiros.

| Busca | Trabalho certo em 1º | Entre os 5 primeiros | Entre os 10 primeiros | MRR |
|---|---|---|---|---|
| Semântica | 20% | 30% | 37% | 0,25 |
| **BM25** | **60%** | **73%** | **75%** | **0,66** |
| Híbrida | 30% | 55% | 67% | 0,41 |

![Avaliação da busca](reports/figures/avaliacao_busca.png)

**O BM25 ganhou com folga, e a busca semântica puxou a híbrida para baixo.** O exemplo do câncer,
em que a híbrida se saiu melhor, era um caso só; com 60 perguntas, o quadro é outro.

A busca semântica foi mal por causa do modelo de embeddings, reaproveitado do projeto de evolução
temática. Lá ele servia para agrupar resumos parecidos entre si. Aqui a tarefa é outra: ligar uma
pergunta curta a um resumo longo. Além disso, ele lê só as primeiras 100 palavras do resumo, e o
título não entra no vetor.

**Limitações:** cada pergunta tem um único trabalho certo, mas outros trabalhos também podem
respondê-la e aqui contam como erro. Com 60 perguntas, cada porcentagem tem margem de cerca de
±12 pontos. E as perguntas, escritas por um LLM a partir do resumo, às vezes repetem termos do
próprio trabalho, o que favorece o BM25.
