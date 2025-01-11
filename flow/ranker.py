import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRanker

# Sample DataFrame
data = pd.DataFrame({
    "week": [1, 1, 1, 2, 2, 2],
    "position": ["C", "C", "C", "C", "C", "C"],
    "player_id": [1, 2, 3, 1, 2, 3],
    "feature1": [10, 20, 15, 12, 22, 18],
    "feature2": [30, 25, 35, 32, 28, 30],
    "rank": [1, 2, 3, 1, 2, 3],  # Lower rank indicates better performance
})

# Encode categorical features
le_position = LabelEncoder()
data["position_encoded"] = le_position.fit_transform(data["position"])

# Feature matrix
X = data[["week", "position_encoded", "feature1", "feature2"]]

# Target vector
y = data["rank"]

# Generate query IDs
data["qid"] = data.groupby(["week", "position"]).ngroup()
qid = data["qid"]

# Split the data
X_train, X_test, y_train, y_test, qid_train, qid_test = train_test_split(X, y, qid, test_size=0.2, random_state=42)

# Initialize the XGBRanker
model = XGBRanker(objective="rank:pairwise", learning_rate=0.1, n_estimators=100, max_depth=6, random_state=42)

# Fit the model
model.fit(
    X_train,
    y_train,
    group=qid_train.groupby(qid_train).size().to_numpy(),
    eval_set=[(X_test, y_test)],
    eval_group=[qid_test.groupby(qid_test).size().to_numpy()],
    # eval_metric="ndcg",
    # early_stopping_rounds=10,
    verbose=True,
)

# Predict
predictions = model.predict(X_test)

# Evaluate
# (Add your evaluation metrics here)
