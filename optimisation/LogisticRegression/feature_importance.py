"""
Plot & Study the feature importance of Logistic Regression.
"""

from cuml import LogisticRegression
import matplotlib.pyplot as plt
from config import *

# Import training data
X_train, Y_train = import_dataset(config["EXPORT"]["l1_feature_eliminated_path"])

def optimal_model():
    return LogisticRegression(C=0.03233598851589464, max_iter=1463, penalty="l1", linesearch_max_iter=161)

# Fit a Logistic Regression L1 model
model = optimal_model()
model.fit(X_train, Y_train)

# Feature Importance by model coefficient
coefficients = [x[0] for x in model.coef_]
feature_importance = sorted(zip(X_train.columns, coefficients), key=lambda v: -abs(v[1]))

# Feature Importance Plot - Top 20
feature_importance = feature_importance[:20]
fig, ax = plt.subplots(figsize=(10, 7))
ax.bar([x for x, y in feature_importance], [abs(y) for x, y in feature_importance])
ax.set_title("Feature Importance (Logistic Regression L1)")
ax.set_xlabel("Feature")
ax.set_ylabel("Model Coefficients (Absolute)")
plt.xticks(rotation=-35, ha="left")
fig.tight_layout()
plt.savefig(f'{config["EXPORT"]["image_path"]}/LogisticRegression_FeatureImportance.png')