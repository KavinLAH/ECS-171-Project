# Data Processing
import numpy as np
import pandas as pd

# RF Model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint

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