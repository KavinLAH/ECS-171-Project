import pandas as pd
import numpy as np
from joblib import load


def main():
    df = pd.read_csv('NBA_testing.csv')
    count = 0
    passed = False
    df["player"] = df["player"].str.lower()
    player_names = df["player"].to_list()
    while count < 3 and passed == False:
        player1 = input("Player 1 name: ")
        if player1.lower() not in player_names:
            print(f"Error: {player1} not found in NBA_testing.csv. Please try again. \nAvailable players include: {player_names[:3]} ...")
        else:
            passed = True
        count += 1
    if count == 3:
        print(f"Error. Bad name entries. Enter a real player name. For example: {player_names[:1]}")
        return
    
    count = 0
    passed = False
    while count < 3 and passed == False:
        player2 = input("Player 2 name: ")
        if player2.lower() not in player_names:
            print(f"Error: {player2} not found in NBA_testing.csv. Please try again.\nAvailable players include: {player_names[:3]} ...")
        else:
            passed = True
        count += 1
    if count == 3:
        print(f"Error. Bad name entries. Enter a real player name. For example: {player_names[:1]}")
        return

    # Load the trained Random Forest model
    model = load('BBALL_RF_mdl.joblib')

    # Load NBA test data
    # Keep 'player' for lookup, drop only for model input
    # 'score' is kept for model input and labeling
    df = df.dropna()
    df_indexed = df.set_index("player")
    df_new =  df_indexed.loc[[player1.lower(), player2.lower()]].reset_index()
    # Drop the 'player' column for model input (keep 'score')
    data = df_new.drop(columns=["player"]).to_numpy()
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

    pred = model.predict(X)[0]
    print(pred)
    winner = player1 if pred == 1 else player2
    print(f"Prediction: {winner} is more likely to win a 1-on-1 matchup between {player1} and {player2}.")

if __name__ == '__main__':
    main() 