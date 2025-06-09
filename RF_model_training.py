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
# UNCOMMENT IF LOOKING TO GET INFORMATION ON FEATURE IMPORTANCE:

# # Feature Importance Plot
# importances = rf.feature_importances_

# # Your 42 feature names, exactly matching len(importances)
# feature_names = [
#  'L_MPG','L_PPG','L_FGM','L_FGA','L_FG%','L_3PM','L_3PA','L_3P%',
#  'L_FTM','L_FTA','L_FT%','L_ORB','L_DRB','L_RPG','L_APG','L_SPG',
#  'L_BPG','L_TOV','L_PF','L_Height','L_Weight',
#  'R_MPG','R_PPG','R_FGM','R_FGA','R_FG%','R_3PM','R_3PA','R_3P%',
#  'R_FTM','R_FTA','R_FT%','R_ORB','R_DRB','R_RPG','R_APG','R_SPG',
#  'R_BPG','R_TOV','R_PF','R_Height','R_Weight'
# ]

# # Sort importances descending
# indices = np.argsort(importances)[::-1]

# # Plot
# plt.figure(figsize=(12, 6))
# plt.bar(np.arange(len(importances)), importances[indices], align='center')
# plt.xticks(np.arange(len(importances)), np.array(feature_names)[indices], rotation=90)
# plt.xlabel('Feature')
# plt.ylabel('Importance')
# plt.title('Random Forest Feature Importances')
# plt.tight_layout()
# # plt.savefig('feature_importance.pdf', format='pdf', bbox_inches='tight')

# # Construct df_box for boxplots
# # comments are non-updated !!!
# # Indices for FG%, Height, and PPG in your data (adjust if needed)
# FGM_idx = 2   # FG% index in your data
# MPG_idx = 0  # Height index in your data
# PPG_idx = 1    # PPG index in your data

# # For each pair, use the left player's features if label==1 (winner), else right player's features
# FGM = np.where(labels == 1, left_rows[:, FGM_idx], right_rows[:, FGM_idx])
# PPG = np.where(labels == 1, left_rows[:, PPG_idx], right_rows[:, PPG_idx])
# MPG = np.where(labels == 1, left_rows[:, MPG_idx], right_rows[:, MPG_idx])

# # For losers, use the other side's features
# FGM_loser = np.where(labels == 0, left_rows[:, FGM_idx], right_rows[:, FGM_idx])
# PPG_loser = np.where(labels == 0, left_rows[:, PPG_idx], right_rows[:, PPG_idx])
# MPG_loser = np.where(labels == 0, left_rows[:, MPG_idx], right_rows[:, MPG_idx])

# # Combine winners and losers into a DataFrame
# FGM_all = np.concatenate([FGM, FGM_loser])
# PPG_all = np.concatenate([PPG, PPG_loser])
# MPG_all = np.concatenate([MPG, MPG_loser])
# label_all = np.concatenate([np.ones_like(labels), np.zeros_like(labels)])

# df_box = pd.DataFrame({'FGM': FGM_all, 'PPG': PPG_all, 'MPG': MPG_all, 'label': np.concatenate([np.ones_like(labels), np.zeros_like(labels)])})
# df_box['Outcome'] = df_box['label'].map({1: 'Winner', 0: 'Loser'})

# # Boxplots for FG%, Height, and PPG between winners and losers

# features = ['FGM', 'PPG', 'MPG']
# fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# for ax, feat in zip(axes, features):
#     sns.boxplot(x='Outcome', y=feat, data=df_box, ax=ax)
#     ax.set_title(f"{feat} by Outcome")
#     ax.set_xlabel('')
#     ax.set_ylabel(feat)

# plt.tight_layout()
# # plt.savefig('predicted_outcome.pdf', bbox_inches='tight')
# plt.show()

