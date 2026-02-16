import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

# ── 1. Load dataset ────────────────────────────────────────────────────────────
data = pd.read_excel("High_Accuracy_Sport_Injury_Dataset.xlsx")
print(f"Dataset loaded: {data.shape[0]} rows, {data.shape[1]} columns")

# ── 2. Features & target ───────────────────────────────────────────────────────
FEATURE_COLS = [
    "Age", "Gender", "Height_cm", "Weight_kg", "BMI",
    "Training_Frequency", "Training_Duration", "Warmup_Time",
    "Sleep_Hours", "Flexibility_Score", "Muscle_Asymmetry",
    "Recovery_Time", "Injury_History", "Stress_Level", "Training_Intensity",
]
TARGET_COL = "Injury_Risk"

X = data[FEATURE_COLS]
y = data[TARGET_COL]

# ── 3. Train / test split ──────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train size: {len(X_train)}  |  Test size: {len(X_test)}")

# ── 4. Hyperparameter tuning with GridSearchCV ─────────────────────────────────
print("\nRunning hyperparameter search (this may take ~30s)...")
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "class_weight": ["balanced", None],
}
base_model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(
    base_model,
    param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1,
    verbose=1,
)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print(f"\nBest parameters: {best_params}")
model = grid_search.best_estimator_

# ── 5. Cross-validation score ──────────────────────────────────────────────────
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1")
print(f"Cross-val F1 scores: {cv_scores.round(3)}")
print(f"Mean CV F1: {cv_scores.mean():.4f}  ±  {cv_scores.std():.4f}")

# ── 6. Evaluation on held-out test set ────────────────────────────────────────
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
roc_auc  = roc_auc_score(y_test, y_prob)

print("\n─── Test Set Evaluation ─────────────────────────────────────────────────")
print(f"Accuracy : {accuracy:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Low Risk", "High Risk"]))
print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
print(f"  FN={cm[1,0]}  TP={cm[1,1]}")

# ── 7. Feature importances ────────────────────────────────────────────────────
print("\nTop Feature Importances:")
importances = pd.Series(model.feature_importances_, index=FEATURE_COLS)
for feat, imp in importances.sort_values(ascending=False).items():
    bar = "█" * int(imp * 50)
    print(f"  {feat:<22} {imp:.4f}  {bar}")

# ── 8. Save model ─────────────────────────────────────────────────────────────
with open("injury_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\n✅  Model saved as injury_model.pkl")
print(f"    Best params: {best_params}")
print(f"    Test Accuracy: {accuracy:.4f}  |  ROC-AUC: {roc_auc:.4f}")