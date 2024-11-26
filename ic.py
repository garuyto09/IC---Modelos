import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, accuracy_score, log_loss, roc_auc_score,
                             classification_report, roc_curve)
from sklearn.inspection import permutation_importance


excel_file_path = 'data.xls'

# # # Read the Excel file into a Pandas DataFrame
df = pd.read_excel(excel_file_path)
df.rename(columns={'default payment next month': 'default'}, inplace=True)
# # Definir X e y
y = df['default']
X = df.drop(['default', 'ID'], axis=1)


# Separando os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Treinando e ajustando os modelos com os melhores valores de hiperparâmetros

# Logistic Regression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Treinando e ajustando os modelos com os melhores valores de hiperparâmetros

# Logistic Regression
logreg = LogisticRegression(solver='newton-cholesky')
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)
y_pred_logreg_proba = logreg.predict_proba(X_test)[:, 1]

# KNN com melhor K
knn = KNeighborsClassifier(n_neighbors=11)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
y_pred_knn_proba = knn.predict_proba(X_test)[:, 1]

# Decision Tree com melhores parâmetros
dt = DecisionTreeClassifier(max_depth=3, min_samples_leaf=1, min_samples_split=2, random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
y_pred_dt_proba = dt.predict_proba(X_test)[:, 1]

# Random Forest com melhores parâmetros
rf = RandomForestClassifier(max_depth=10, min_samples_leaf=1, min_samples_split=10, n_estimators=50, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_pred_rf_proba = rf.predict_proba(X_test)[:, 1]

# Função para calcular e imprimir as métricas
def print_metrics(y_true, y_pred, y_pred_proba):
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Log Loss:", log_loss(y_true, y_pred_proba))
    print("AUC-ROC:", roc_auc_score(y_true, y_pred_proba))
    print("Classification Report:\n", classification_report(y_true, y_pred))

# Métricas dos modelos
print("Modelo: Logistic Regression")
print_metrics(y_test, y_pred_logreg, y_pred_logreg_proba)

print("\nModelo: KNN")
print_metrics(y_test, y_pred_knn, y_pred_knn_proba)

print("\nModelo: Decision Tree")
print_metrics(y_test, y_pred_dt, y_pred_dt_proba)

print("\nModelo: Random Forest")
print_metrics(y_test, y_pred_rf, y_pred_rf_proba)

# Função para plotar a matriz de confusão em uma figura compartilhada
def plot_confusion_matrices(y_true_list, y_pred_list, titles):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))  # Configura a figura com 4 subplots
    axes = axes.flatten()  # Flatten para facilitar a iteração
    for i, (y_true, y_pred, title) in enumerate(zip(y_true_list, y_pred_list, titles)):
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[i],
                    annot_kws={"size": 16},  # Aumenta o tamanho dos números
                    xticklabels=['0', '1'], yticklabels=['0', '1'])
        axes[i].set_title(f'Matriz de Confusão: {title}', fontsize=18)  # Título maior
        axes[i].set_xlabel('Predicted', fontsize=14)  # Aumenta a fonte dos eixos
        axes[i].set_ylabel('Actual', fontsize=14)
        axes[i].tick_params(axis='both', which='major', labelsize=12)  # Aumenta os ticks

    plt.tight_layout()
    plt.show()

# Listas de dados para as matrizes de confusão
y_true_list = [y_test, y_test, y_test, y_test]
y_pred_list = [y_pred_logreg, y_pred_knn, y_pred_dt, y_pred_rf]
titles = ['Logistic Regression', 'KNN', 'Decision Tree', 'Random Forest']

# Plotar todas as matrizes de confusão com fonte maior
plot_confusion_matrices(y_true_list, y_pred_list, titles)


# Função para plotar a curva ROC
def plot_roc_curve(fpr, tpr, title, color):
    plt.plot(fpr, tpr, linestyle='-', color=color, label=title, linewidth=2)  


# Curvas ROC
fpr_logreg, tpr_logreg, _ = roc_curve(y_test, y_pred_logreg_proba)
fpr_knn, tpr_knn, _ = roc_curve(y_test, y_pred_knn_proba)
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_pred_dt_proba)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf_proba)

# Ajustando os estilos de linha para serem mais consistentes
plot_roc_curve(fpr_logreg, tpr_logreg, 'Logistic Regression', 'blue')
plot_roc_curve(fpr_knn, tpr_knn, 'KNN', 'orange')
plot_roc_curve(fpr_dt, tpr_dt, 'Decision Tree', 'green')
plot_roc_curve(fpr_rf, tpr_rf, 'Random Forest', 'red')


plt.xlabel('False Positive Rate', fontsize=15)
plt.ylabel('True Positive Rate', fontsize=15)
plt.title('Curva ROC', fontsize=15)
plt.legend(loc='lower right', fontsize=15)  # Aumenta o tamanho da fonte da legenda
plt.grid(True)
plt.show()

#######Importância das variáveis para cada modelo
def plot_feature_importance(model, model_name, feature_names, subplot_position):
    if model_name == 'Regressão Logística':
        importance = np.abs(model.coef_[0])
    else:
        importance = model.feature_importances_
    
    indices = np.argsort(importance)[::-1]
    plt.subplot(subplot_position)
    plt.title(f'Importância das Variáveis - {model_name}')
    plt.bar(range(X.shape[1]), importance[indices], align='center')
    plt.xticks(range(X.shape[1]), feature_names[indices], rotation=90)
    plt.xlim([-1, X.shape[1]])

# Importância das variáveis para KNN usando permutação
perm_importance_knn = permutation_importance(knn, X_test, y_test, n_repeats=10, random_state=42)
perm_sorted_idx_knn = perm_importance_knn.importances_mean.argsort()[::-1]

plt.figure(figsize=(15, 10))

plot_feature_importance(logreg, 'Regressão Logística', X.columns, 221)
plt.subplot(222)
plt.title('Importância das Variáveis - KNN')
plt.bar(range(X.shape[1]), perm_importance_knn.importances_mean[perm_sorted_idx_knn], align='center')
plt.xticks(range(X.shape[1]), X.columns[perm_sorted_idx_knn], rotation=90)
plt.xlim([-1, X.shape[1]])

plot_feature_importance(dt, 'Árvore de Decisão', X.columns, 223)
plot_feature_importance(rf, 'Floresta Aleatória', X.columns, 224)

plt.tight_layout()
plt.show()
