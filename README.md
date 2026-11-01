Projeto de estudo de Machine Learning (ML) para classificação de textos (sentimentos), realizado durante a disciplina de TAIA (Tópicos Avançados em IA), ministrado pelo prof. Caio Libânio na UFRPE-UAST.

# Classificador de Fake News com Pré-processamento de Texto

Este repositório contém um notebook em Python para construir um classificador de notícias falsas utilizando técnicas clássicas de Processamento de Linguagem Natural (NLP) e Machine Learning. O objetivo principal é transformar textos em representações numéricas úteis, treinar modelos de classificação e avaliar a capacidade de distinguir entre notícias reais e falsas.

## Objetivo do projeto

O projeto segue um fluxo típico de aprendizado supervisionado para classificação de textos:

1. Carregar um conjunto de dados contendo notícias e rótulos;
2. Unir o título e o texto em uma única coluna textual;
3. Pré-processar o texto para reduzir ruído e melhorar a qualidade dos dados;
4. Converter textos em vetores numéricos;
5. Treinar diferentes modelos de classificação;
6. Avaliar seu desempenho com métricas quantitativas.

Esse tipo de solução é muito comum em tarefas de detecção automática de desinformação, análise de conteúdo textual e sistemas de classificação de documentos.

## Estrutura do repositório

- `classificador_fake_news_pre_processed_by_text.ipynb` — notebook principal com todo o pipeline de treinamento e avaliação;
- `README.md` — documentação do projeto.

## Dataset

O notebook usa um arquivo CSV armazenado no Google Drive:

- `/content/drive/MyDrive/Tópicos Avançados em IA/dataset_fn.csv`

A estrutura esperada do dataset inclui colunas como:

- `title` — título da notícia;
- `text` — texto completo da notícia;
- `label` — rótulo da notícia (`REAL` ou `FAKE`).

Durante o processamento, os rótulos são convertidos para valores binários:

- `REAL` -> `1`
- `FAKE` -> `0`

Também é criada uma coluna `news` concatenando título e texto para formar uma base textual única por notícia.

## Pipeline do notebook

### 1. Importação de bibliotecas

O notebook usa várias bibliotecas fundamentais para análise de dados e NLP:

- `pandas` — leitura e manipulação de dados tabulares;
- `scikit-learn` — divisão dos dados, vetorização, treinamento e métricas;
- `nltk` — download e uso de stopwords em inglês;
- `spacy` — lematização do texto em inglês;
- `matplotlib` e `seaborn` — visualização de resultados e matrizes de confusão.

### 2. Pré-processamento do texto

A limpeza do texto é um passo essencial para reduzir ruído e melhorar a qualidade das features. O notebook realiza:

- conversão para letras minúsculas;
- remoção de caracteres especiais e dígitos com `re.sub`;
- remoção de stopwords;
- lematização via spaCy.

A função `limpa_texto()` encapsula esse fluxo e transforma cada notícia em uma versão mais padronizada e útil para o modelo.

### 3. Vetorização

Algoritmos de aprendizado de máquina não entendem texto puro, então o conteúdo precisa ser convertido em representação numérica. Neste projeto, são comparadas duas formas principais:

- `CountVectorizer`: conta a frequência de cada termo no texto;
- `TfidfVectorizer`: atribui pesos conforme a importância do termo no documento e no conjunto.

Esses vetorizadores geram matrizes esparsas que servem de entrada para os classificadores.

### 4. Divisão treino/teste

O notebook usa `train_test_split` para separar o conjunto em treinamento e teste:

- `test_size=0.3`
- `random_state=13`

Isso permite avaliar a capacidade do modelo de generalizar para dados que não foram vistos durante o treinamento.

### 5. Modelos treinados

Diferentes classificadores são testados para comparar desempenho:

- `LogisticRegression`;
- `MultinomialNB` (Naive Bayes Multinomial);
- `MLPClassifier` (Perceptron de múltiplas camadas).

O objetivo é verificar qual combinação entre vetorizador e classificador apresenta melhor performance.

### 6. Métricas de avaliação

O notebook usa métricas clássicas para verificar a qualidade do modelo:

- `accuracy_score` — acurácia;
- `roc_auc_score` — área sob a curva ROC;
- `f1_score` — F1-score ponderado;
- `confusion_matrix` — matriz de confusão.

Além disso, a função `mostra_metricas()` exibe os valores das métricas e gera um heatmap da matriz de confusão com `seaborn`.

## Técnicas e conceitos aplicados

Este projeto reúne conceitos fundamentais de NLP e Machine Learning:

- pré-processamento textual;
- limpeza e normalização de texto;
- stopwords;
- lematização;
- tokenização e vetorização;
- aprendizado supervisionado;
- divisão treino/teste;
- comparação entre modelos;
- avaliação por métricas de classificação.

## Bibliotecas utilizadas

### Bibliotecas principais

- `pandas`
- `numpy`
- `re`
- `matplotlib`
- `seaborn`
- `nltk`
- `spacy`
- `scikit-learn`

### Módulos e classes de uso direto

- `train_test_split`
- `CountVectorizer`
- `TfidfVectorizer`
- `LogisticRegression`
- `MultinomialNB`
- `MLPClassifier`
- `accuracy_score`
- `f1_score`
- `roc_auc_score`
- `confusion_matrix`

## Observações sobre o desempenho

O notebook compara vários modelos e combinações de representações textuais. Os resultados mostram que algumas combinações performam melhor do que outras, e a qualidade do pré-processamento impacta diretamente a acurácia e o F1-score.

O projeto também destaca uma abordagem didática muito útil para entender como classificação textual funciona na prática, com foco em:

- representação do texto em números;
- uso de aprendizado supervisionado;
- comparação de modelos;
- escolha de melhores métricas para avaliação.

## Como executar

### Google Colab

1. Abra o notebook no Google Colab;
2. Monte o Google Drive;
3. Ajuste o caminho do CSV, se necessário;
4. Execute as células em ordem.
