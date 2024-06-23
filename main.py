import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

from model_trainer import train_and_evaluate_classifiers
from model_evaluator import plot_roc_curves

def main():
    path = 'Projeto2_IA_UnB_2024_1/archive/LLM.csv'

    # Treina e avalia os modelos
    results = train_and_evaluate_classifiers(path)

    # Exibe os resultados
    for name, result in results.items():
        print(f"{name}: AUC = {result['auc']:.2f}")

    # Lança as curvas ROC e salva imagem
    plot_roc_curves(results, 'Projeto2_IA_UnB_2024_1/results/roc_curves.png')

if __name__ == '__main__':
    main()
