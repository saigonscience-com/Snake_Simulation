import json

class DataLogger:
    def __init__(self, log_file="training_data.json"):
        self.log_file = log_file
        self.data = []

    def log(self, episode, score, epsilon):
        self.data.append({
            "episode": episode,
            "score": score,
            "epsilon": epsilon
        })

    def save(self):
        with open(self.log_file, "w") as f:
            json.dump(self.data, f, indent=4)
        print(f"Data saved to {self.log_file}")

    def load(self):
        try:
            with open(self.log_file, "r") as f:
                self.data = json.load(f)
            print(f"Data loaded from {self.log_file}")
        except FileNotFoundError:
            print(f"No data file found: {self.log_file}")

