# Data Processing
import numpy as np
import pandas as pd

# LG Model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
import matplotlib.pyplot as plt
import seaborn as sns

from joblib import dump
# import graphviz

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
lg = LogisticRegression(max_iter=1000, class_weight='balanced')
lg.fit(X_train, y_train)

accuracy = lg.score(X_test, y_test)
print("Test accuracy:", accuracy)

dump(lg, 'BBALL_LG_mdl.joblib')

# # FEATURE IMPORTANCE CODE
# # get feature importance through magnitude
# coefs       = lg.coef_.ravel()      
# importances = np.abs(coefs)             

# # rename because currently in numpy names
# feature_names = [
#  'L_MPG','L_PPG','L_FGM','L_FGA','L_FG%','L_3PM','L_3PA','L_3P%',
#  'L_FTM','L_FTA','L_FT%','L_ORB','L_DRB','L_RPG','L_APG','L_SPG',
#  'L_BPG','L_TOV','L_PF','L_Height','L_Weight',
#  'R_MPG','R_PPG','R_FGM','R_FGA','R_FG%','R_3PM','R_3PA','R_3P%',
#  'R_FTM','R_FTA','R_FT%','R_ORB','R_DRB','R_RPG','R_APG','R_SPG',
#  'R_BPG','R_TOV','R_PF','R_Height','R_Weight'
# ]

# indices = np.argsort(importances)[::-1]


# # split between winners and losers
# FGM_idx = 2   # FGM column
# PPG_idx = 1   # PPG column
# MPG_idx = 0   # MPG column

# FGM       = np.where(y==1, left_rows[:,FGM_idx], right_rows[:,FGM_idx])
# PPG       = np.where(y==1, left_rows[:,PPG_idx], right_rows[:,PPG_idx])
# MPG       = np.where(y==1, left_rows[:,MPG_idx], right_rows[:,MPG_idx])

# FGM_loser = np.where(y==0, left_rows[:,FGM_idx], right_rows[:,FGM_idx])
# PPG_loser = np.where(y==0, left_rows[:,PPG_idx], right_rows[:,PPG_idx])
# MPG_loser = np.where(y==0, left_rows[:,MPG_idx], right_rows[:,MPG_idx])

# # set up box plots
# df_box = pd.DataFrame({
#     'FGM': np.concatenate([FGM, FGM_loser]),
#     'PPG': np.concatenate([PPG, PPG_loser]),
#     'MPG': np.concatenate([MPG, MPG_loser]),
#     'Outcome': ['Winner']*len(y) + ['Loser']*len(y)
# })

# fig, axes = plt.subplots(1, 3, figsize=(18, 6))
# for ax, feat in zip(axes, ['FGM','PPG','MPG']):
#     sns.boxplot(x='Outcome', y=feat, data=df_box, ax=ax)
#     ax.set_title(f"{feat} by Outcome")
#     ax.set_xlabel('')
#     ax.set_ylabel(feat)

# plt.tight_layout()
# plt.savefig('LG_pred_outcome.pdf', bbox_inches='tight')
# plt.show()
