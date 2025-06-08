# compare_roc_curves.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from joblib import load

# Load test data
# Use the same processing as in LG_on_NBA_data_test.py

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

# Get predicted probabilities for the positive class
lg_probs = lg.predict_proba(X_test)[:, 1]
rf_probs = rf.predict_proba(X_test)[:, 1]
gnb_probs = gnb.predict_proba(X_test)[:, 1]

# Compute ROC curve and AUC for each model
fpr_lg, tpr_lg, _ = roc_curve(y_true, lg_probs)
fpr_rf, tpr_rf, _ = roc_curve(y_true, rf_probs)
fpr_gnb, tpr_gnb, _ = roc_curve(y_true, gnb_probs)
auc_lg = auc(fpr_lg, tpr_lg)
auc_rf = auc(fpr_rf, tpr_rf)
auc_gnb = auc(fpr_gnb, tpr_gnb)

# Plot ROC curves
plt.figure(figsize=(8, 6))
plt.plot(fpr_lg, tpr_lg, label=f'Logistic Regression (AUC = {auc_lg:.2f})')
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {auc_rf:.2f})')
plt.plot(fpr_gnb, tpr_gnb, label=f'GNB (AUC = {auc_gnb:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Chance')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()
