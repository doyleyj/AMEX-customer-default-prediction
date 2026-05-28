"""
Plot & Study the feature importance of Random Forest Classifier.
"""

from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from config import *

# Import training data
X_train, Y_train = import_dataset(f"{config['EXPORT']['feature_eliminated_path']}/selected_features", eliminate=False, standardize=False)

def optimal_model():
    return RandomForestClassifier(n_estimators=927, max_depth=77, min_samples_split=8, min_samples_leaf=4, n_jobs=-1)

# Fit a Logistic Regression L1 model
model = optimal_model()
model.fit(X_train, Y_train)

# Feature Importance by model coefficient
feature_importance = model.feature_importances_
sorted_feature_importances = sorted(zip(X_train.columns, feature_importance), reverse=True, key=lambda v: v[1])

# Feature Importance Plot - Top 20
sorted_feature_importances = sorted_feature_importances[:20]
x = [x for x, _ in sorted_feature_importances]
y = [y for _, y in sorted_feature_importances]
fig, ax = plt.subplots(figsize=(10, 7))
ax.bar(x, y)
plt.xticks(rotation=-35, ha="left")
ax.set_title("Feature Importance (Random Forest)")
ax.set_xlabel("Feature")
ax.set_ylabel("Gini Importance")
fig.tight_layout()
plt.savefig(f'{config["EXPORT"]["image_path"]}/RandomForest_FeatureImportance.png')