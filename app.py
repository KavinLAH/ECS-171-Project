from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from joblib import load

app = Flask(__name__)

df = pd.read_csv('NBA_testing.csv').dropna()
df['player'] = df['player'].str.lower()
model = load('BBALL_RF_mdl.joblib')

@app.route('/', methods=['GET', 'POST'])
def index():
    error = None
    prediction = None

    if request.method == 'POST':
        player1 = request.form.get('player1', '').strip().lower()
        player2 = request.form.get('player2', '').strip().lower()

        player_names = df['player'].to_list()
        if player1 not in player_names or player2 not in player_names:
            error = (
                f"Error: one or both names not found. "
                f"Examples: {player_names[:3]}"
            )
        else:
            df_indexed = df.set_index('player')
            df_new = df_indexed.loc[[player1, player2]].reset_index()

            data = df_new.drop(columns=['player']).to_numpy()
            col_idx = data.shape[1] - 1

            n = data.shape[0]
            i_idx, j_idx = np.triu_indices(n, k=1)
            left_rows  = data[i_idx]
            right_rows = data[j_idx]

            labels = (left_rows[:, col_idx] > right_rows[:, col_idx]).astype(int)
            matrix = np.hstack((left_rows, right_rows, labels.reshape(-1,1)))

            c = data.shape[1]
            drop_cols = [col_idx, col_idx + c]
            m_red = np.delete(matrix, drop_cols, axis=1)
            X = m_red[:, :-1]

            pred = model.predict(X)[0]
            winner = player1 if pred == 1 else player2
            prediction = (
                f"Prediction: {winner.title()} is more likely to win a 1-on-1 matchup "
                f"between {player1.title()} and {player2.title()}."
            )

    return render_template('index.html',
                           error=error,
                           prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
