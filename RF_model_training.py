# Data Processing
import numpy as np
import pandas as pd

# RF Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint

from joblib import dump
import matplotlib.pyplot as plt
import seaborn as sns

# DATA PROCESSING
# create matrix and assign winner
print("loading data")
data = np.loadtxt('Training_Data.csv', delimiter = ',', skiprows=1, usecols=range(1, 23))

col_idx = 21 # compare score volume of each player (last column, 0-based)

# generate all pair combinations of indices in the csv (comparing each player)
n = data.shape[0]
i_idx, j_idx = np.triu_indices(n, k = 1)

left_rows = data[i_idx]
right_rows = data[j_idx]

# assigns score to each combination of players.
labels = (left_rows[:,col_idx] > right_rows[:, col_idx]).astype(int)
print("assigned labels")

# creates matrix with left and right information of each player.
matrix = np.hstack((
    left_rows,
    right_rows,
    labels.reshape(-1,1)
))
print("matrix shape:", matrix.shape)

# drop score column because i dont want it as a feature. two score columns, one for each player. drop the left one, then go through all the columns and drop the other one (+c)
c = data.shape[1]
drop_cols = [col_idx, col_idx + c]


m_red = np.delete(matrix, drop_cols, axis=1)

X = m_red[:, :-1]
y = m_red[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)

# FITTING MODEL
print("training")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

accuracy = rf.score(X_test, y_test)
print("Test accuracy:", accuracy)

dump(rf, 'BBALL_RF_mdl.joblib')

# Feature Importance Plot
importances = rf.feature_importances_
# Feature names: left and right features, excluding dropped score columns
feature_names = []
for side in ['L', 'R']:
    for i in range(data.shape[1]):
        if i != col_idx:
            feature_names.append(f'{side}{i}')
plt.figure(figsize=(12, 6))
indices = np.argsort(importances)[::-1]
plt.bar(range(len(importances)), importances[indices], align='center')
plt.xticks(range(len(importances)), np.array(feature_names)[indices], rotation=90)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Random Forest Feature Importances')
plt.tight_layout()
plt.show()

# Construct df_box for boxplots
# Indices for FG%, Height, and PPG in your data (adjust if needed)
FG_idx = 7   # FG% index in your data
Height_idx = 2  # Height index in your data
PPG_idx = 17    # PPG index in your data

# For each pair, use the left player's features if label==1 (winner), else right player's features
FG_percent = np.where(labels == 1, left_rows[:, FG_idx], right_rows[:, FG_idx])
Height = np.where(labels == 1, left_rows[:, Height_idx], right_rows[:, Height_idx])
PPG = np.where(labels == 1, left_rows[:, PPG_idx], right_rows[:, PPG_idx])

# For losers, use the other side's features
FG_percent_loser = np.where(labels == 0, left_rows[:, FG_idx], right_rows[:, FG_idx])
Height_loser = np.where(labels == 0, left_rows[:, Height_idx], right_rows[:, Height_idx])
PPG_loser = np.where(labels == 0, left_rows[:, PPG_idx], right_rows[:, PPG_idx])

# Combine winners and losers into a DataFrame
FG_all = np.concatenate([FG_percent, FG_percent_loser])
Height_all = np.concatenate([Height, Height_loser])
PPG_all = np.concatenate([PPG, PPG_loser])
label_all = np.concatenate([np.ones_like(labels), np.zeros_like(labels)])

df_box = pd.DataFrame({'FG%': FG_all, 'Height': Height_all, 'PPG': PPG_all, 'label': label_all})

# Boxplots for FG%, Height, and PPG between winners and losers

features = ['FG%', 'Height', 'PPG']
labels = {0: 'Loser', 1: 'Winner'}

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for idx, feature in enumerate(features):
    sns.boxplot(x='label', y=feature, data=df_box, ax=axes[idx])
    axes[idx].set_xticklabels([labels.get(int(l.get_text()), l.get_text()) for l in axes[idx].get_xticklabels()])
    axes[idx].set_xlabel('Predicted Outcome')
    axes[idx].set_ylabel(feature)
    axes[idx].set_title(f'{feature} by Predicted Outcome')
plt.tight_layout()
plt.show()