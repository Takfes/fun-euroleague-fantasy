import pandas as pd


# Functional API
def add_ratio_feature(data, column_a, column_b, new_column_name):
    data[new_column_name] = data[column_a] / data[column_b]
    return data


class FeatureEngineer:
    def __init__(self, data):
        self.data = data.copy()
        self.history = []

    def add(self, func, **kwargs):
        self.data = func(self.data, **kwargs)
        self.history.append(f"Applied {func.__name__} with {kwargs}")
        return self

    def add_ratio_feature(self, column_a, column_b, new_column_name):
        return self.add(add_ratio_feature, column_a=column_a, column_b=column_b, new_column_name=new_column_name)

    @staticmethod
    def validate_column(data, column_name):
        if column_name not in data.columns:
            raise ValueError(f"Column {column_name} not found.")

    @property
    def columns(self):
        return self.data.columns.tolist()
