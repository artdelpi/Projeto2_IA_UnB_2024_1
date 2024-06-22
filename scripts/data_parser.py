import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix

def parse_data(path:str) -> tuple[csr_matrix, csr_matrix, pd.core.series.Series, pd.core.series.Series]:
    """
    Carrega dados do arquivo CSV a partir do path fornecido,
    separa atributos de rótulos, divide dados para teste  e  treinamento e
    vectoriza textos para serem usados por algoritmos de machine learning.
    
    Params:
    - path (str): O caminho para o arquivo CSV.

    Returns:
    - texts_train_vec (csr_matrix): Textos vectorizados para treinamento.
    - texts_test_vec (csr_matrix): Textos vectorizados para teste.
    - labels_train (Series): Rótulos de treinamento.
    - labels_test (Series): Rótulos de teste.
    """
    # Carrega dados do arquivo CSV
    try:
        file = pd.read_csv(path)
    except FileNotFoundError:
        raise FileNotFoundError(f'Arquivo {path} não encontrado.')
    except pd.errors.ParserError:
        raise ValueError(f'Não foi possível analisar o arquivo {path}')


    # Separa atributos de rótulos
    file.dropna(subset=['Text', 'Label'], inplace=True)
    texts, labels = file['Text'], file['Label']

    # Separa dados para teste e treinamento
    texts_train, texts_test, labels_train, labels_test = train_test_split(
        texts, labels, 
        test_size=0.3, 
        random_state=42
    )
    
    # Vectoriza os textos
    vectorizer = TfidfVectorizer()
    texts_train_vec = vectorizer.fit_transform(texts_train)
    texts_test_vec = vectorizer.transform(texts_test)

    return texts_train_vec, texts_test_vec, labels_train, labels_test
