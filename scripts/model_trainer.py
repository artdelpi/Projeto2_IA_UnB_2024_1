import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from data_parser import parse_data

def train_and_evaluate_classifiers(path):

    # Parse dos dados
    texts_train_vec, texts_test_vec, labels_train, labels_test = parse_data(path)

    # Verifica e remove valores ausentes
    labels_train.dropna(inplace=True)
    labels_test.dropna(inplace=True)

    # Converte rótulos para valores binários
    labels_train_binary = labels_train.apply(lambda x: 1 if x == 'ai' else 0)
    labels_test_binary = labels_test.apply(lambda x: 1 if x == 'ai' else 0)

    classifiers = {
        'LDA': LDA(),
        'QDA': QDA(),
        'K-NN': KNeighborsClassifier(),
        'SVM': SVC(probability=True),
        'Random Forest': RandomForestClassifier()
    }

    results = {}

    for name, clf in classifiers.items():
        if name in ['LDA', 'QDA']:
            # Converte para arrays densos
            texts_train_dense = texts_train_vec.toarray()
            texts_test_dense = texts_test_vec.toarray()
            clf.fit(texts_train_dense, labels_train_binary)
            y_pred_proba = clf.predict_proba(texts_test_dense)[:, 1]
        else:
            clf.fit(texts_train_vec, labels_train_binary)
            y_pred_proba = clf.predict_proba(texts_test_vec)[:, 1]

        auc_score = roc_auc_score(labels_test_binary, y_pred_proba)
        fpr, tpr, _ = roc_curve(labels_test_binary, y_pred_proba)

        results[name] = {
            'classifier': clf,
            'auc': auc_score,
            'fpr': fpr,
            'tpr': tpr
        }

    return results
