## Your 1v1 NBA Player Predictor

There are many players in the NBA who thrive in 5-on-5 play, but far fewer have tested their skills in 1-on-1 matchups. This project aims to explore:
Can we use traditional 5-on-5 statistics and player attributes to predict the outcome of a 1-on-1 matchup between any two basketball players?

Given the lack of real 1-on-1 data, we train a model using Euroleague statistics and evaluate it on NBA players to see how well the model generalizes. Our features include performance stats (PTS, REB, STL, etc.) and physical attributes (height, weight) to create a predictive pipeline. 

### How To Run
1. Create and activate a virtual environment (optional but highly recommended)
2. Run 'pip install -r requirements.txt'
3. Run 'python LogReg_model_training.py'
4. Run 'python app.py'
5. Open your browser on your device. Type in 'http://127.0.0.1:5000' or the endpoint address that is shown in the line where it says "* Running on {endpoint} into the search tab" if it does not match and press enter. It should open the UI to use our system.
6. Enter player names into UI. If you are interested which players you can run, take a look at our 'NBA_testing.csv', although most current/past NBA players should be accessible.

## Project Description
Who would win in a 1-on-1 basketball game — LeBron James or Stephen Curry?
This question sparks endless debates among fans, analysts, and players alike. But what if we could bring data-driven insights into the conversation?

This project uses machine learning to predict the winner of a hypothetical 1v1 basketball matchup between NBA players. By analyzing key performance metrics from real-world 5v5 games, our model estimates head-to-head outcomes based on player strengths, weaknesses, and statistical tendencies.

Built as a project for UC Davis’s ECS 171 Machine Learning course, this project explores the complexities of adapting 5v5 basketball data to simulate 1v1 scenarios — a task that requires thorough feature selection, data engineering, and model tuning.

Whether you're a sports analyst, a data science enthusiast, or just a basketball fan with strong opinions, this project brings advanced analytics to an ongoing debate in a creative way.

# Key Highlights:
--> Real-world datasets from Euroleague (training) and NBA (testing)
--> Support for multiple ML algorithms (Decision Tree, Random Forest, XGBoost)
--> Feature-rich input: scoring, playmaking, shooting efficiency, size, defense
--> Clean CLI for easy model training and prediction between any two players