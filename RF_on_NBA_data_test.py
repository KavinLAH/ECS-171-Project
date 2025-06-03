from joblib import load
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

# load previously trained RF model
model = load('BBALL_RF_mdl.joblib')

df = pd.read_csv("NBA_testing.csv")
df = df.dropna()

data = df.to_numpy()

# data = np.loadtxt('NBA_testing.csv', delimiter = ',', skiprows=1)

col_idx = 22 # compare score volume of each player

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
print(m_red.shape)

X_test = m_red[:, :-1]
y_true = m_red[:, -1]

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_true, y_pred)
print("Test accuracy:", accuracy)