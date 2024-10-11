import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


class DataCollector:
    def __init__(self):
        self.data = []  # To store data from each snake's life

    def log_data(self, state, action, outcome):
        # Collect state and action (features), and the outcome (target)
        features = state + [action]
        self.data.append((features, outcome))

    def save_data(self, filename="snake_data.csv"):
        # Save the data to a CSV file for future use
        df = pd.DataFrame(self.data, columns=["State", "Outcome"])
        df.to_csv(filename, index=False)

    def train_model(self):
        # Train a linear regression model on the collected data
        if len(self.data) < 10:  # Check if we have enough data
            return None

        X = np.array([d[0] for d in self.data])  # Features (state + action)
        y = np.array([d[1] for d in self.data])  # Targets (outcomes)

        model = LinearRegression()
        model.fit(X, y)
        return model
