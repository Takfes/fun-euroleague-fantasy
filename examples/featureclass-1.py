import pandas as pd


# Functional API Example
def add_ratio_feature(data, column_a, column_b, new_column_name):
    data[new_column_name] = data[column_a] / data[column_b]
    return data


def add_rolling_mean(data, column, window, new_column_name):
    data[new_column_name] = data[column].rolling(window).mean()
    return data


class FeatureEngineer:
    def __init__(self, data):
        self.data = data.copy()
        self.history = []  # Stores the series of operations

    def add(self, func, **kwargs):
        """
        Apply a transformation and record the operation in history.
        """
        # Apply the function
        self.data = func(self.data, **kwargs)

        # Record the operation
        self.history.append((func, kwargs))
        return self  # Enable chaining

    def replay(self, data=None):
        """
        Reapply the recorded transformations to a new DataFrame.
        """
        # Start with the provided data or the original
        data_to_transform = data.copy() if data is not None else self.data.copy()

        # Replay all operations
        for func, kwargs in self.history:
            data_to_transform = func(data_to_transform, **kwargs)

        return data_to_transform

    def persist_blueprint(self, filepath):
        """
        Persist the blueprint of operations as a serialized file.
        """
        import pickle

        with open(filepath, "wb") as f:
            pickle.dump(self.history, f)

    def load_blueprint(self, filepath):
        """
        Load a previously saved blueprint.
        """
        import pickle

        with open(filepath, "rb") as f:
            self.history = pickle.load(f)


# Example Usage
if __name__ == "__main__":
    # Sample dataset
    df = pd.DataFrame({"A": range(1, 11), "B": range(11, 21)})

    # Create a FeatureEngineer instance and chain transformations
    fe = FeatureEngineer(df)
    fe.add(add_ratio_feature, column_a="A", column_b="B", new_column_name="A_B_ratio").add(
        add_rolling_mean, column="A", window=3, new_column_name="A_rolling_3"
    )

    # Persist the blueprint
    fe.persist_blueprint("blueprint.pkl")

    # Reapply the blueprint to a new dataset
    new_df = pd.DataFrame({"A": range(11, 21), "B": range(21, 31)})
    reloaded_fe = FeatureEngineer(new_df)
    reloaded_fe.load_blueprint("blueprint.pkl")
    transformed_data = reloaded_fe.replay()

    print(transformed_data)
