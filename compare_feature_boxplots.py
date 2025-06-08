import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load test data (same processing as compare_roc_curves.py)
df = pd.read_csv("NBA_testing.csv").dropna()
data = df.drop(columns=["player"]).to_numpy()
col_idx = data.shape[1] - 1
n = data.shape[0]
i_idx, j_idx = np.triu_indices(n, k=1)
left_rows = data[i_idx]
right_rows = data[j_idx]
labels = (left_rows[:, col_idx] > right_rows[:, col_idx]).astype(int)

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
feature_labels = ['Normalized FG%', 'Standardized Height', 'Normalized PPG']
label_names = {0: 'Loser', 1: 'Winner'}

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for idx, feature in enumerate(features):
    sns.boxplot(x='label', y=feature, data=df_box, ax=axes[idx])
    axes[idx].set_xticklabels([label_names.get(int(l.get_text()), l.get_text()) for l in axes[idx].get_xticklabels()])
    axes[idx].set_xlabel('Predicted Outcome')
    axes[idx].set_ylabel(feature_labels[idx])
    axes[idx].set_title(f'{feature_labels[idx]} by Predicted Outcome')
plt.tight_layout()
plt.show()
