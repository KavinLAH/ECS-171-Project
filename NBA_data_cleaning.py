import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df_totals = pd.read_csv('Player Totals.csv')


cols_to_keep = ["lg", # league, to be dropped later
                "player", # to be dropped
                "season", # to be dropped
                "mp", # not per game
                "g", 
                "pts", # not per game
                "fg", # not per game
                "fga", # not per game
                "fg_percent", 
                "x3p", # not per game
                "x3pa", # not per game
                "x3p_percent",
                "ft",# not per game
                "fta",# not per game
                "ft_percent",
                "orb", # not per game
                "drb", # not per game
                "trb", # not per game
                "ast", # not per game
                "stl", # not per game
                "blk", # not per game
                "tov", # not per game
                "pf", # not per game
                ]
"""
mp: minuts played
g: games played
pts: total points scored
fg: volume scoring, total field goals made
fga: fg attempted
fg_percent: fg / fga
x3p: # 3pointers made
x3pa: 3 pointers attempted
x3p_percent: x3p / x3pa
ft: free throws made
fta: free throws attempted
ft_percent: ft / fta
orb: offensive rebounds
drb: def reb
trb: orb + drb
ast: assists
stl: steals
blk: blocks
tov: turnovers
pf: personal fouls
"""


columns_to_drop_totals = [col for col in df_totals.columns if col not in cols_to_keep]
df_totals.drop(columns=columns_to_drop_totals, inplace=True)

# Filter out ABA league
df_totals = df_totals[df_totals['lg'] != 'ABA']

# Keep only the most recent season stat line for each player
df_totals = df_totals.sort_values(by=['player', 'season'], ascending=[True, False])
df_totals = df_totals.drop_duplicates(subset='player', keep='first')

# Drop rows with any NaN in the remaining Player Totals columns
df_totals = df_totals.dropna()
print(f"Processed Player Totals data: {df_totals.shape[0]} players")

exclude = ["lg", "player", "season", "fg_percent", "x3p_percent", "ft_percent"]

cols_mod = [col for col in df_totals if col not in exclude]
df_totals[cols_mod] = df_totals[cols_mod].div(df_totals["g"], axis=0)

# Load and Preprocess Draft Combine Data 
print("\nLoading and preprocessing Draft Combine data...")
df_combine = pd.read_csv('draft_combine_train.csv')

# Drop specified columns from Draft Combine
columns_to_drop_combine = [
    "yearDraft", "yearCombine", "numberPickOverall", "position", "drafted", 
    "wingspan", "reach_standing", "standing_vertical", "max_vertical", "bench_reps",
    "timeLaneAgility", "timeThreeQuarterCourtSprint", "timeModifiedLaneAgility",
    "lengthHandInches", "widthHandInches", "body_fat_pct"
]
df_combine.drop(columns=columns_to_drop_combine, inplace=True)

# Rename player_name to player for merging
df_combine.rename(columns={'player_name': 'player'}, inplace=True)

# Identify combine measurement columns (before dropping player_id)
combine_measurement_cols = [col for col in df_combine.columns if col not in ['player_id', 'player']]
# Drop player_id as we are merging on name 'player'
df_combine.drop(columns=['player_id'], inplace=True)

print(f"Processed Draft Combine data: {df_combine.shape[0]} entries")

# Merge the two datasets 
print("\nMerging Player Totals and Draft Combine data...")
# Use an inner merge to keep only players present in both datasets
df_merged = pd.merge(df_totals, df_combine, on='player', how='inner')
print(f"Initial merged data: {df_merged.shape[0]} players")

# Handle Missing Combine Data in Merged Set 
print(f"Checking for rows missing *all* combine measurements ({combine_measurement_cols})...")
initial_rows = df_merged.shape[0]
# Drop rows where ALL combine measurement columns are NaN
df_merged.dropna(subset=combine_measurement_cols, how='all', inplace=True)
rows_dropped = initial_rows - df_merged.shape[0]
print(f"Dropped {rows_dropped} players who had no combine measurements recorded.")
print(f"Final merged dataset contains {df_merged.shape[0]} players.")

df_merged.drop(columns=["lg", "season", "g"], inplace=True)

print(f"# of columns in merged_df: {df_merged.columns}")



rename_map = {
    'mp': 'MPG',
    'pts': 'PPG',
    'fg': 'FGM',
    'fga': 'FGA',
    'fg_percent': 'FG%',
    'x3p': '3PM',
    'x3pa': '3PA',
    'x3p_percent': '3P%',
    'ft': 'FTM',
    'fta': 'FTA',
    'ft_percent': 'FT%',
    'orb': 'ORB',
    'drb': 'DRB',
    'trb': 'RPG',
    'ast': 'APG',
    'stl': 'SPG',
    'blk': 'BPG',
    'tov': 'TOV',
    'pf': 'PF',
    'height': 'Height',
    'weight': 'Weight'
}

df = df_merged.rename(columns=rename_map)
print(df.head())

data_characteristics = {}

groups = {
    "Game & Scoring": ['MPG', 'PPG', 'FGM', 'FGA', 'FG%', '3PM', '3PA', '3P%', 'FTM', 'FTA', 'FT%'],
    "Rebounding & Playmaking": ['ORB', 'DRB', 'RPG', 'APG'],
    "Defense & Turnovers": ['SPG', 'BPG', 'TOV', 'PF'],
    "Physical Attributes": ['Height', 'Weight']
}

# plot each group in a grid
for group_name, features in groups.items():
    n_cols = 3
    n_rows = (len(features) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    fig.suptitle(f'{group_name} Distributions', fontsize=16)
    axes = axes.flatten()

    for i, col in enumerate(features):
        # make plot
        sns.histplot(df[col], kde=True, bins=30, ax=axes[i])
        axes[i].set_title(col)

        # save mean, stddev, median, IQR of feature
        data_characteristics[col] = {
            "mean": df[col].mean(), 
            "stddev": df[col].std(), 
            "median": df[col].median(), 
            "iqr": df[col].quantile(0.75)-df[col].quantile(0.25)
        }

    # Turn off any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

weights = {
    'MPG': 0.3,     # Small signal of coach trust, not much for 1v1. mp column
    'PPG': 1.0,     # Strongest weight – scoring is everything take # of games and convert to pts/game
    'FGM': 0.8,     # Volume scoring # fg
    'FGA': 0.6,     # Volume attempts – shows aggressiveness #fga
    'FG%': 0.9,     # Efficient shooting is key #fg_percent
    '3PM': 0.7,     # Making threes gives space #x3p
    '3PA': 0.5,     # Attempts show range
    '3P%': 0.8,     # Accuracy from 3 is crucial
    'FTM': 0.1,     # Barely matters
    'FTA': 0.1,     # Same
    'FT%': 0.2,     # Same
    'ORB': 0.4,     # Second chances
    'DRB': 0.4,     # Defensive control
    'RPG': 0.5,     # Overall rebounding ability
    'APG': 0.3,     # Passing not supefr relevant in 1v1, but can indicate ball-handling ability, IQ
    'SPG': 0.6,     # Shows defensive instincts
    'BPG': 0.7,     # Rim protection matters in 1v1
    'TOV': 0.1,     # Not very useful, but a very turnover-prone player might be exposed
    'PF': 0.0,      # Not used in 1v1
    'Height': 0.8,  # Major advantage in post or defense
    'Weight': 0.6   # Strength helps to hold position or bully
}


scaled_features = {}

for feature, w in weights.items():
    print(feature)
    print(w)
    # Get mean and stddev from the data_characteristics dictionary (created by Melody)
    mean = data_characteristics[feature]['mean']
    stddev = data_characteristics[feature]['stddev']

    # Avoid division by zero if stddev is zero (rare but possible)
    if stddev == 0:
        print(f"Warning: stddev for {feature} is zero, skipping scaling.")
        scaled = df[feature] * 0  # all zeros (no variation)
    else:
        # Calculate Z-score using precomputed mean and stddev
        z_score = (df[feature] - mean) / stddev

        # Multiply by weight to scale influence (0 to 1)
        scaled = z_score * w

    # Store the scaled feature with suffix '_scaled'
    scaled_features[f"{feature}_scaled"] = scaled

# Create a DataFrame of all players using containing all scaled features
df_scaled = pd.DataFrame(scaled_features)

df_scaled["score"] = df_scaled.sum(axis=1)

# Add player names to the scaled DataFrame as the first column
if "player" in df.columns:
    df_scaled["player"] = df["player"].values
    # Move 'player' column to the front
    ordered_cols = ["player",
        'MPG_scaled', 'PPG_scaled', 'FGM_scaled', 'FGA_scaled', 'FG%_scaled', '3PM_scaled', '3PA_scaled', '3P%_scaled',
        'FTM_scaled', 'FTA_scaled', 'FT%_scaled', 'ORB_scaled', 'DRB_scaled', 'RPG_scaled', 'APG_scaled', 'SPG_scaled',
        'BPG_scaled', 'TOV_scaled', 'PF_scaled', 'Height_scaled', 'Weight_scaled', 'score']
    df_scaled = df_scaled[ordered_cols]
else:
    print("Warning: 'player' column not found in df. Player names will not be included in NBA_testing.csv.")

df_scaled.to_csv("NBA_testing.csv", index=False)