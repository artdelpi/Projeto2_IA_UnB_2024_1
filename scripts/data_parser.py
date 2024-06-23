import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from numpy import ndarray

def parse_data(data_path:str) -> tuple[ndarray, ndarray, pd.core.series.Series, pd.core.series.Series]:
    try:
        data_file = pd.read_csv(data_path)
    except FileNotFoundError as e:
        print(f'Erro: O arquivo {data_path} não foi encontrado.')
        raise(e)
    except pd.errors.EmptyDataError as e:
        print(f'Erro: O arquivo {data_path} está vazio.')
        raise(e)
    except pd.erros.ParserError as e:
        print(f'Erro: Ocorreu um erro ao parsear o arquivo {data_path}')
        raise(e)
    except Exception as e:
        print(f'Erro: Ocorreu um erro inesperado ao ler o arquivo {data_path}')
        raise(e)
    
    # Verifica estrutura esperada do DataFrame ('Text'|'Label')
    if 'Text' not in data_file.columns or 'Label' not in data_file.columns:
        raise ValueError('Erro: As colunas "Text" e/ou "Label" não foram encontradas.')

    # Vectoriza os textos
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data_file['Text']).toarray()
    y = data_file['Label'].apply(lambda x:1 if x == 'ai' else 0)

    # Separa dados pra teste (0.3) e treinamento (0.7)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test
