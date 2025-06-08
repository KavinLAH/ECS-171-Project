import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load
from sklearn.metrics import accuracy_score

# Load test data (same processing as compare_roc_curves.py)
df = pd.read_csv("NBA_testing.csv").dropna()
data = df.drop(columns=["player"]).to_numpy()
col_idx = data.shape[1] - 1
n = data.shape[0]
i_idx, j_idx = np.triu_indices(n, k=1)
left_rows = data[i_idx]
right_rows = data[j_idx]
labels = (left_rows[:, col_idx] > right_rows[:, col_idx]).astype(int)
matrix = np.hstack((left_rows, right_rows, labels.reshape(-1, 1)))
c = data.shape[1]
drop_cols = [col_idx, col_idx + c]
m_red = np.delete(matrix, drop_cols, axis=1)
X_test = m_red[:, :-1]
y_true = m_red[:, -1]

# Load models
lg = load('BBALL_LG_mdl.joblib')
rf = load('BBALL_RF_mdl.joblib')
gnb = load('BBALL_GNB_mdl.joblib')

# Get predictions
lg_pred = lg.predict(X_test)
rf_pred = rf.predict(X_test)
gnb_pred = gnb.predict(X_test)

# Compute accuracies
acc_lg = accuracy_score(y_true, lg_pred)
acc_rf = accuracy_score(y_true, rf_pred)
acc_gnb = accuracy_score(y_true, gnb_pred)

# Plot bar chart
models = ['Logistic Regression', 'Random Forest', 'Gaussian NB']
accuracies = [acc_lg, acc_rf, acc_gnb]
plt.figure(figsize=(8, 6))
plt.bar(models, accuracies, color=['blue', 'green', 'orange'])
plt.ylim(0, 1)
plt.ylabel('Test Accuracy')
plt.title('Test Accuracy Comparison of Models')
for i, v in enumerate(accuracies):
    plt.text(i, v + 0.01, f"{v*100:.2f}%", ha='center', fontweight='bold')
plt.tight_layout()
plt.show()
