import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score
from data_parser import parse_data
from sklearn import metrics

def train_and_evaluate_classifiers(data_path):

    # Pré-processa conjunto de dados
    X_train, X_test, y_train, y_test = parse_data(data_path)

    classifiers = {
        'LDA': LDA(),
        'QDA': QDA(),
        'KNN': KNN(),
        'SVM': SVC(probability=True),
        'Random Forest': RandomForestClassifier()
    }

    results = {}

    for name, clf in classifiers.items():
        # Treina o modelo
        clf.fit(X_train, y_train)

        # Prediciona probabilidades
        y_proba = clf.predict_proba(X_test)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)

        # Prediciona rótulos
        y_pred = clf.predict(X_test)

        # Calcula métricas de desempenho
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        
        results[name] = {
            'fpr': fpr,
            'tpr': tpr,
            'auc': roc_auc,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall
        }

    return results
