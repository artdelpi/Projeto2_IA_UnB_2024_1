# Projeto2_IA_UnB_2024_1

- **Link do repositório no GitHub**: https://github.com/artdelpi/Projeto2_IA_UnB_2024_1.git

## Algoritmos de Aprendizagem Supervisionada para Classificação de Textos Gerados por IA ou Humanos

Este projeto visa analisar os resultados obtidos na classificação de dados gerados por IA ou humanos, ao utilizar diferentes algoritmos de classificação supervisionada (LDA, QDA, KNN, SVM e Random Forest).

### Descrição dos Modelos Utilizados

- **Linear Discriminant Analysis (LDA)**: Opera buscando uma combinação linear de características que separa duas ou mais classes de dados. Assume que as variáveis preditoras têm a mesma matriz de covariância. É eficaz para dados lineares.

- **Quadratic Discriminant Analysis (QDA)**: Similar ao LDA, porém cada classe tem sua própria matriz de covariância. É eficaz para dados lineares.

- **K-Nearest Neighbors (KNN)**: Classifica baseado nos k vizinhos mais próximos no espaço de características. É eficaz para dados não-lineares.

- **Support Vector Machine (SVM)**: Busca um hiperplano no espaço de características que maximiza a margem entre dados de classificações diferentes. É eficaz para dados não-lineares.

- **Random Forest**: Gera várias árvores de decisão, a partir de conjuntos de dados bootstrap e utiliza a média das predições das árvores. Evita overfitting e é eficaz para dados não-lineares.

### Resultados Obtidos

- **LDA (AUC = 0.81)**
- **QDA (AUC = 0.84)**
- **KNN (AUC = 0.99)**
- **SVM (AUC = 1.00)**
- **Random Forest (AUC = 1.00)**

![Gráfico da Curva ROC](Projeto2_IA_UnB_2024_1/results/roc_curves.png)

### Métricas Utilizadas

Para avaliar o desempenho dos modelos, utilizou-se o score AUC e curva ROC. 

- **AUC**: Métrica que vai de 0 (erro total de classificação) a 1 (acerto total de classificação), sendo 0.5 equivalente ao acaso. Obtida, em código, via chamada da função "auc()" da biblioteca sklearn, passando os argumentos FPR e TPR.

- **ROC**: Linha que relaciona valores de TPR (Taxa de Verdadeiro Positivo) e FPR (Taxa de Falso Positivo) pra cada modelo. Essas taxas foram obtidas comparando as predições de classificação com os rótulos esperados. Obtida, em código, via chamada da função "roc_curve()", passando os argumentos y_test e y_proba.

### Interpretação dos Resultados

Os modelos KNN, SVM e Random Forest apresentaram o melhor desempenho de classificação, enquanto LDA e QDA tiveram desempenho razoável. 

Os motivos por trás da perfomance do KNN, SVM e Random Forest são:

1. Não-linearidade dos dados: Problemas que envolvem texto tendem a ter natureza não-linear ao serem vetorizados usando TF-IDF.
    - **TF-IDF (Term Frequency-Inverse Document Frequency)**: É uma técnica de vetorização de texto que transforma textos em números. Busca destacar palavras importantes e reduzir a importância de palavras comuns.

2. Complexidade dos textos: O conteúdo de cada entrada a ser classificada pode envolver grande quantidade de características/nuances, que fazem com que o texto de IA e humano sejam diferentes em várias dimensões, favorecendo os modelos não-lineares. 

Por extensão, uma linha reta não é tão eficiente para separar os dados de diferentes classificação, tal qual uma curva. Afinal, aferiu-se que os modelos que aproximam a função hipótese para classificação de dados não-lineares performaram melhor.

### Estrutura do Projeto Explicada e Observação

Ao executar o arquivo main.py, os dados serão:
1. Pré-processados (data_parser.py),
2. Modelos serão treinados e AUC e ROC obtidos (model_trainer.py)
3. Imagem da curva ROC será plotada (model_evaluator.py) 

- **Obs**: Espera-se que ao executar a main.py, seja gerada a imagem "roc_curves.png", caso seja excluida, e que os scores AUC de cada modelo seja impresso no painel de OUTPUT.

### Como Rodar o Projeto

1. Ter o Python 3 instalado.
2. Instalar as dependências listadas no arquivo `requirements.txt`:
   ```bash
   pip install -r requirements.txt