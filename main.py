import pandas as pd
import numpy as np
from joblib import load
from sklearn.metrics import confusion_matrix, classification_report

# Load the trained Random Forest model
model = load('BBALL_RF_mdl.joblib')

# Load NBA test data
# Keep 'player' for lookup, drop only for model input
# 'score' is kept for model input and labeling

df = pd.read_csv('NBA_testing.csv')
df = df.dropna()

# Drop the 'player' column for model input (keep 'score')
data = df.drop(columns=["player"]).to_numpy()
col_idx = data.shape[1] - 1  # Index of the score column (last column)

# Generate all pairwise combinations of players
n = data.shape[0]
i_idx, j_idx = np.triu_indices(n, k=1)

left_rows = data[i_idx]
right_rows = data[j_idx]

# Assign labels: 1 if left player score > right player score, else 0
labels = (left_rows[:, col_idx] > right_rows[:, col_idx]).astype(int)

# Create matrix with left and right player features and label
matrix = np.hstack((left_rows, right_rows, labels.reshape(-1, 1)))

# Drop the score columns (left and right)
c = data.shape[1]
drop_cols = [col_idx, col_idx + c]
m_red = np.delete(matrix, drop_cols, axis=1)

X = m_red[:, :-1]
y = m_red[:, -1]

# Before fitting the model, make sure X has the same columns/features as used during training.
# For example, if you are using a DataFrame, ensure you are not accidentally including extra columns.

# Example fix:
# Suppose 'X' is your features DataFrame and you want only the 40 features used for training:
# Replace this line:
# model.fit(X, y)
# With something like:

expected_features = [
    # List the 40 feature names in the exact order used for training
    # e.g. 'MPG', 'PPG', ..., 'Weight'
]
X = X[expected_features]

# Now fit and score as usual
model.fit(X, y)
accuracy = model.score(X, y)

# Test the model
accuracy = model.score(X, y)
print(f"Test accuracy: {accuracy:.4f}")

# Optionally, print confusion matrix and other metrics
preds = model.predict(X)
print("Confusion Matrix:\n", confusion_matrix(y, preds))
print("Classification Report:\n", classification_report(y, preds))

# --- Compare two NBA players by name ---
print("\n--- 1-on-1 NBA Player Comparison ---")
player_names = df["player"].values

# Example test case: pick two players from the dataset
player1 = "LeBron James"
player2 = "Stephen Curry"

# Find the rows for each player
row1 = df[df["player"] == player1]
row2 = df[df["player"] == player2]

if row1.empty or row2.empty:
    print(f"Error: One or both players not found in NBA_testing.csv.\nAvailable players include: {player_names[:10]} ...")
else:
    # Remove 'player' column for model input (keep 'score')
    features1 = row1.drop(["player"], axis=1).values[0]
    features2 = row2.drop(["player"], axis=1).values[0]
    # Stack features as model expects: [player1_features, player2_features]
    pair_features = np.hstack([features1, features2]).reshape(1, -1)
    pred = model.predict(pair_features)[0]
    winner = player1 if pred == 1 else player2
    print(f"Prediction: {winner} is more likely to win a 1-on-1 matchup between {player1} and {player2}.")