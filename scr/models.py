from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from preprocess import preprocess
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


sns.set_theme(style='whitegrid')


def ensure_result_dir():
    res_dir = Path(__file__).resolve().parents[1] / 'result'
    res_dir.mkdir(parents=True, exist_ok=True)
    return res_dir


def train_and_evaluate(save_plots: bool = True):
    res_dir = ensure_result_dir()

    # Carrega dados pré-processados
    data = preprocess()

    x_train = data['x_train_scaled']
    x_test = data['x_test_scaled']
    x_train_smote = data['x_train_smote']
    y_train = data['y_train']
    y_test = data['y_test']
    y_train_smote = data['y_train_smote']
    feature_names = data['feature_names']

    # Random Forest with GridSearch (baseline)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [5, 8, None],
        'min_samples_leaf': [1, 3],
    }

    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=7, class_weight='balanced'),
        param_grid, cv=5, scoring='f1', n_jobs=-1
    )
    grid_search.fit(x_train, y_train)
    print('Melhores parâmetros (RF):', grid_search.best_params_)
    rf = grid_search.best_estimator_
    rf.fit(x_train, y_train)
    y_pred_rf = rf.predict(x_test)
    y_proba_rf = rf.predict_proba(x_test)[:, 1]

    print('\n=== Random Forest ===')
    print('Acurácia:', accuracy_score(y_test, y_pred_rf))
    print(classification_report(y_test, y_pred_rf, target_names=['Baixa/Média', 'Alta Qualidade']))
    print('AUC-ROC:', roc_auc_score(y_test, y_proba_rf))

    if save_plots:
        # Feature importance
        importancias = pd.Series(rf.feature_importances_, index=feature_names).sort_values()
        fig, ax = plt.subplots(figsize=(8, 6))
        importancias.plot(kind='barh', ax=ax, color='steelblue')
        ax.set_title('Feature Importance — Random Forest')
        fig.tight_layout()
        fig.savefig(res_dir / 'rf_feature_importance.png')
        plt.close(fig)

    # Decision Tree baseline
    dt = DecisionTreeClassifier(random_state=7, criterion='entropy', max_depth=3)
    dt.fit(x_train, y_train)
    y_pred_dt = dt.predict(x_test)
    y_proba_dt = dt.predict_proba(x_test)[:, 1]

    print('\n=== Decision Tree ===')
    print('Acurácia:', accuracy_score(y_test, y_pred_dt))
    print(classification_report(y_test, y_pred_dt, target_names=['Baixa/Média', 'Alta Qualidade']))
    print('AUC-ROC:', roc_auc_score(y_test, y_proba_dt))

    # KNN: find best k
    k_vals = range(1, 21)
    accuracies = []
    for k in k_vals:
        knn_temp = KNeighborsClassifier(n_neighbors=k)
        knn_temp.fit(x_train, y_train)
        accuracies.append(accuracy_score(y_test, knn_temp.predict(x_test)))

    best_k = k_vals[int(np.argmax(accuracies))]
    print(f"\nMelhor K (baseline): {best_k} (Acurácia: {max(accuracies):.4f})")

    knn = KNeighborsClassifier(n_neighbors=best_k)
    knn.fit(x_train, y_train)
    y_pred_knn = knn.predict(x_test)
    y_proba_knn = knn.predict_proba(x_test)[:, 1]

    print('\n=== KNN ===')
    print('Acurácia:', accuracy_score(y_test, y_pred_knn))
    print(classification_report(y_test, y_pred_knn, target_names=['Baixa/Média', 'Alta Qualidade']))
    print('AUC-ROC:', roc_auc_score(y_test, y_proba_knn))

    # Logistic Regression
    lr = LogisticRegression(random_state=7, max_iter=1000, class_weight='balanced')
    lr.fit(x_train, y_train)
    y_pred_lr = lr.predict(x_test)
    y_proba_lr = lr.predict_proba(x_test)[:, 1]

    print('\n=== Regressão Logística ===')
    print('Acurácia:', accuracy_score(y_test, y_pred_lr))
    print(classification_report(y_test, y_pred_lr, target_names=['Baixa/Média', 'Alta Qualidade']))
    print('AUC-ROC:', roc_auc_score(y_test, y_proba_lr))

    # Salvar gráficos individuais dos modelos (ROC e Matriz de Confusão)
    if save_plots:
        # Decision Tree - ROC
        fpr_dt, tpr_dt, _ = roc_curve(y_test, y_proba_dt)
        auc_dt = roc_auc_score(y_test, y_proba_dt)
        plt.figure(figsize=(7, 5))
        plt.plot(fpr_dt, tpr_dt, color='steelblue', linewidth=2, label=f'Decision Tree (AUC = {auc_dt:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
        plt.xlabel('Taxa de Falsos Positivos')
        plt.ylabel('Taxa de Verdadeiros Positivos')
        plt.title('Curva ROC — Decision Tree')
        plt.legend()
        plt.tight_layout()
        plt.savefig(res_dir / 'roc_decision_tree.png')
        plt.close()

        # Decision Tree - Confusion
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_dt), display_labels=['Baixa/Média', 'Alta Qualidade']).plot(ax=ax, colorbar=False, cmap='Blues')
        ax.set_title('Matriz de Confusão — Decision Tree')
        plt.tight_layout()
        fig.savefig(res_dir / 'confusion_dt.png')
        plt.close(fig)

        # Random Forest - ROC
        fpr_rf_i, tpr_rf_i, _ = roc_curve(y_test, y_proba_rf)
        auc_rf_i = roc_auc_score(y_test, y_proba_rf)
        plt.figure(figsize=(7, 5))
        plt.plot(fpr_rf_i, tpr_rf_i, color='#2ca02c', linewidth=2, label=f'Random Forest (AUC = {auc_rf_i:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
        plt.xlabel('Taxa de Falsos Positivos')
        plt.ylabel('Taxa de Verdadeiros Positivos')
        plt.title('Curva ROC — Random Forest')
        plt.legend()
        plt.tight_layout()
        plt.savefig(res_dir / 'roc_random_forest.png')
        plt.close()

        # Random Forest - Confusion
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_rf), display_labels=['Baixa/Média', 'Alta Qualidade']).plot(ax=ax, colorbar=False, cmap='Greens')
        ax.set_title('Matriz de Confusão — Random Forest')
        plt.tight_layout()
        fig.savefig(res_dir / 'confusion_rf.png')
        plt.close(fig)

        # KNN - ROC
        fpr_knn_i, tpr_knn_i, _ = roc_curve(y_test, y_proba_knn)
        auc_knn_i = roc_auc_score(y_test, y_proba_knn)
        plt.figure(figsize=(7, 5))
        plt.plot(fpr_knn_i, tpr_knn_i, color='#9467bd', linewidth=2, label=f'KNN (AUC = {auc_knn_i:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
        plt.xlabel('Taxa de Falsos Positivos')
        plt.ylabel('Taxa de Verdadeiros Positivos')
        plt.title('Curva ROC — KNN')
        plt.legend()
        plt.tight_layout()
        plt.savefig(res_dir / 'roc_knn.png')
        plt.close()

        # KNN - Confusion
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_knn), display_labels=['Baixa/Média', 'Alta Qualidade']).plot(ax=ax, colorbar=False, cmap='Blues')
        ax.set_title('Matriz de Confusão — KNN')
        plt.tight_layout()
        fig.savefig(res_dir / 'confusion_knn.png')
        plt.close(fig)

        # Logistic Regression - ROC
        fpr_lr_i, tpr_lr_i, _ = roc_curve(y_test, y_proba_lr)
        auc_lr_i = roc_auc_score(y_test, y_proba_lr)
        plt.figure(figsize=(7, 5))
        plt.plot(fpr_lr_i, tpr_lr_i, color='#ff7f0e', linewidth=2, label=f'Regressão Logística (AUC = {auc_lr_i:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
        plt.xlabel('Taxa de Falsos Positivos')
        plt.ylabel('Taxa de Verdadeiros Positivos')
        plt.title('Curva ROC — Regressão Logística')
        plt.legend()
        plt.tight_layout()
        plt.savefig(res_dir / 'roc_logistic_regression.png')
        plt.close()

        # Logistic Regression - Confusion
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_lr), display_labels=['Baixa/Média', 'Alta Qualidade']).plot(ax=ax, colorbar=False, cmap='Oranges')
        ax.set_title('Matriz de Confusão — Regressão Logística')
        plt.tight_layout()
        fig.savefig(res_dir / 'confusion_lr.png')
        plt.close(fig)

    # --- Modelos treinados com SMOTE ---
    print('\n=== Treinando modelos com SMOTE (treino balanceado) ===')

    dt_smote = DecisionTreeClassifier(random_state=7, criterion='entropy', max_depth=3)
    dt_smote.fit(x_train_smote, y_train_smote)
    print('\nDecision Tree + SMOTE')
    print(classification_report(y_test, dt_smote.predict(x_test), target_names=['Baixa/Média', 'Alta Qualidade']))

    rf_smote = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=7,
    )
    rf_smote.fit(x_train_smote, y_train_smote)
    print('\nRandom Forest + SMOTE')
    print(classification_report(y_test, rf_smote.predict(x_test), target_names=['Baixa/Média', 'Alta Qualidade']))

    knn_smote = KNeighborsClassifier(n_neighbors=best_k)
    knn_smote.fit(x_train_smote, y_train_smote)
    print('\nKNN + SMOTE')
    print(classification_report(y_test, knn_smote.predict(x_test), target_names=['Baixa/Média', 'Alta Qualidade']))

    lr_smote = LogisticRegression(random_state=7, max_iter=1000)
    lr_smote.fit(x_train_smote, y_train_smote)
    print('\nRegressão Logística + SMOTE')
    print(classification_report(y_test, lr_smote.predict(x_test), target_names=['Baixa/Média', 'Alta Qualidade']))

    # Probabilidades dos modelos treinados com SMOTE (para comparação)
    y_proba_dt_smote = dt_smote.predict_proba(x_test)[:, 1]
    y_proba_rf_smote = rf_smote.predict_proba(x_test)[:, 1]
    y_proba_knn_smote = knn_smote.predict_proba(x_test)[:, 1]
    y_proba_lr_smote = lr_smote.predict_proba(x_test)[:, 1]

    # Plots comparativos: ROC (baseline vs SMOTE) e AUC em barras
    if save_plots:
        # Salvar ROC de comparação igual ao notebook (baseline)
        plt.figure(figsize=(9, 6))
        for name, proba, color in [
            ('Decision Tree', y_proba_dt, '#1f77b4'),
            ('Random Forest', y_proba_rf, '#2ca02c'),
            ('KNN', y_proba_knn, '#9467bd'),
            ('Logistic Regression', y_proba_lr, '#ff7f0e'),
        ]:
            fpr_i, tpr_i, _ = roc_curve(y_test, proba)
            auc_i = roc_auc_score(y_test, proba)
            plt.plot(fpr_i, tpr_i, color=color, linewidth=2, label=f'{name} (AUC = {auc_i:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
        plt.xlabel('Taxa de Falsos Positivos')
        plt.ylabel('Taxa de Verdadeiros Positivos')
        plt.title('Curva ROC — Comparação dos Modelos')
        plt.legend()
        plt.tight_layout()
        plt.savefig(res_dir / 'roc_comparacao.png')
        plt.close()

        # ROC comparação: baseline (linha contínua) vs SMOTE (linha tracejada)
        plt.figure(figsize=(10, 7))
        models_info = [
            ('Decision Tree', y_proba_dt, y_proba_dt_smote),
            ('Random Forest', y_proba_rf, y_proba_rf_smote),
            ('KNN', y_proba_knn, y_proba_knn_smote),
            ('Logistic Regression', y_proba_lr, y_proba_lr_smote),
        ]
        for name, proba_base, proba_sm in models_info:
            fpr_b, tpr_b, _ = roc_curve(y_test, proba_base)
            auc_b = roc_auc_score(y_test, proba_base)
            plt.plot(fpr_b, tpr_b, linewidth=2, label=f'{name} (Base AUC={auc_b:.3f})')

            fpr_s, tpr_s, _ = roc_curve(y_test, proba_sm)
            auc_s = roc_auc_score(y_test, proba_sm)
            plt.plot(fpr_s, tpr_s, linestyle='--', linewidth=2, label=f'{name} + SMOTE (AUC={auc_s:.3f})')

        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.6)
        plt.xlabel('Taxa de Falsos Positivos')
        plt.ylabel('Taxa de Verdadeiros Positivos')
        plt.title('Curva ROC — Baseline vs SMOTE (todos os modelos)')
        plt.legend(fontsize='small')
        plt.tight_layout()
        plt.savefig(res_dir / 'roc_comparacao_smote.png')
        plt.close()

        # AUC comparativo em barras
        aucs_base = [roc_auc_score(y_test, p) for _, p, _ in models_info]
        aucs_sm = [roc_auc_score(y_test, sm) for _, _, sm in models_info]
        df_auc = pd.DataFrame({'Baseline': aucs_base, 'SMOTE': aucs_sm}, index=[m[0] for m in models_info])
        ax = df_auc.plot(kind='bar', figsize=(9, 6), rot=0, color=['#4C72B0', '#DD8452'])
        ax.set_ylabel('AUC-ROC')
        ax.set_title('AUC-ROC: Baseline vs SMOTE por modelo')
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.3f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='bottom', fontsize=9, xytext=(0, 4), textcoords='offset points')
        plt.tight_layout()
        plt.savefig(res_dir / 'auc_comparacao_smote.png')
        plt.close()


if __name__ == '__main__':
    print('Executando treino e avaliação dos modelos (salvando gráficos em result/)...')
    train_and_evaluate()
