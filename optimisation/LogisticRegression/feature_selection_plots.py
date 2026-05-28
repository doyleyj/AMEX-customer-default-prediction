"""
Plot L1 Feature Elimination and Score Accuracy using the fold informations.
"""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from feature_selection_extract import extract_fold_information
from config import *

c_blue = "#1f77b4"
c_orange = "#ff7f0e"
c_gray = "#7f7f7f"
fold_infos, avg_accuracy, avg_feature_eliminated, argmax_acc = extract_fold_information()
x_grid = fold_infos[0]["c_grid"]

# ------------------------------------------------------------------ #
#                          Plot All 5 folds.                         #
# ------------------------------------------------------------------ #
fit, ax = plt.subplots(figsize=(10, 8))
ax2 = ax.twinx()

for fold in fold_infos:
    ax.plot(x_grid, fold["accuracy"], color=c_blue)
    ax2.plot(x_grid, fold["num_feature_eliminated"], color=c_orange)
    
ax.set_title("Feature Selection with Logistic Regression L1 (5 Folds)")
ax.set_ylabel("Test Accuracy")
ax2.set_ylabel("Number of Features Eliminated")
ax.set_xlabel("L1 Regularization Strength (C)")
legend = [Line2D([0], [0], color=c_blue, lw=2), Line2D([0], [0], color=c_orange, lw=2)]
ax.legend(legend, ["Accuracy", "Features Eliminated"], loc="center right")
plt.savefig(f"{config['EXPORT']['image_path']}/FeatureSelection_NestedCV.png")

# ------------------------------------------------------------------ #
#                      Plot Averaged of 5 Folds.                     #
# ------------------------------------------------------------------ #
fit, ax = plt.subplots(figsize=(10, 8))
ax2 = ax.twinx()

ax.annotate(f"Best Accuracy ({round(avg_accuracy[argmax_acc], 3)})", 
        xy=(x_grid[argmax_acc], avg_accuracy[argmax_acc]),
        xytext=(40, -30), color=c_gray,
        textcoords="offset points", ha="center", va="bottom",
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.3", color=c_gray))
ax.plot(x_grid[argmax_acc], avg_accuracy[argmax_acc], marker="o", color=c_gray, markersize=4)

ax2.annotate(f"Features Eliminated ({avg_feature_eliminated[argmax_acc]})",
        xy=(x_grid[argmax_acc], avg_feature_eliminated[argmax_acc]),
        xytext=(55, 20), color=c_gray,
        textcoords="offset points", ha="center", va="bottom",
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.3", color=c_gray))
ax2.plot(x_grid[argmax_acc], avg_feature_eliminated[argmax_acc], marker="o", color=c_gray, markersize=4)

ax.plot(x_grid, avg_accuracy, color=c_blue)
ax2.plot(x_grid, avg_feature_eliminated, color=c_orange)

ax.set_title("Feature Selection with Logistic Regression L1 (Averaged over 5 folds)")
ax.set_ylabel("Test Accuracy (Averaged)")
ax2.set_ylabel("Number of Features Eliminated (Averaged)")
ax.set_xlabel("L1 Regularization Strength (C)")
legend = [Line2D([0], [0], color=c_blue, lw=2), Line2D([0], [0], color=c_orange, lw=2)]
ax.legend(legend, ["Accuracy", "Features Eliminated"], loc="center right")
plt.savefig(f"{config['EXPORT']['image_path']}/FeatureSelection_AvgFold.png")