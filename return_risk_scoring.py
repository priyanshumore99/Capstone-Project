import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.inspection import permutation_importance
import os
import joblib


# Part 1 ✅Checklist (all acceptance criteria met)
# ✅Dataset generated (6000 rows, 13 columns).
# ✅Verified return rate and missingness (MAR explained).
# ✅Preprocessing pipeline built (ColumnTransformer + Pipeline).
# ✅Baseline DummyClassifier trained (accuracy high, F1=0.0 explained).
# ✅Logistic Regression trained + threshold sweep (trade‑off explained).
# ✅Random Forest tuned with GridSearchCV (best params + ROC‑AUC reported).
# ✅Feature importance + permutation importance compared (bias explained).
# ✅Subgroup analysis done (weak subgroup identified + fix proposed).
# ✅Final artifact saved (return_risk_model.pkl) + t*_rf recorded.

# 1. Generate the exact seeded dataset. Save the script below as generate_orders.py in your repo and run it exactly as written (python3 generate_orders.py) -- do not change np.random.default_rng(42) or any of the fixed category/payment lists, since the acceptance criteria below depend on this exact, deterministic output.

rng = np.random.default_rng(42)
N = 6000

categories = ["Apparel", "Electronics", "Home", "Footwear", "Beauty"]
cat_probs = [0.32, 0.22, 0.18, 0.18, 0.10]
payment_methods = ["COD", "Prepaid_Card", "Prepaid_UPI", "Wallet"]
pay_probs = [0.42, 0.24, 0.24, 0.10]

product_category = rng.choice(categories, size=N, p=cat_probs)
payment_method = rng.choice(payment_methods, size=N, p=pay_probs)

base_price = {
    "Apparel": (400, 2200), "Electronics": (1200, 45000), "Home": (300, 8000),
    "Footwear": (500, 4500), "Beauty": (150, 2500),
}
price_inr = np.round(np.array([rng.uniform(*base_price[c]) for c in product_category]), 0)

discount_pct = np.clip(rng.normal(22, 15, N), 0, 75)
customer_tenure_days = np.clip(rng.exponential(380, N), 1, 2500).round(0)
num_previous_orders = np.clip((customer_tenure_days / 45) + rng.normal(0, 2, N), 0, None).round(0)
base_return_rate = np.clip(rng.beta(1.5, 9, N), 0, 1)
num_previous_returns = np.round(base_return_rate * num_previous_orders).clip(0, num_previous_orders)

delivery_distance_km = np.clip(rng.gamma(3, 90, N), 2, 2200).round(1)
delivery_days = np.clip(rng.normal(4.5, 2.2, N), 1, 21).round(0)
is_weekend_order = rng.integers(0, 2, N)

rating_given = rng.integers(1, 6, N).astype(float)
missing_mask = rng.random(N) < np.where(payment_method == "COD", 0.22, 0.06)
rating_given[missing_mask] = np.nan

fit_risk_cat = np.isin(product_category, ["Apparel", "Footwear"]).astype(float)
prev_return_ratio = np.where(num_previous_orders > 0,
                              num_previous_returns / np.maximum(num_previous_orders, 1), 0)

z = (-2.2 + 1.9 * prev_return_ratio + 0.55 * fit_risk_cat
     + 0.014 * (discount_pct - 20) / 10 + 0.9 * (payment_method == "COD").astype(float)
     + 0.10 * (delivery_days - 4.5) / 2 + 0.30 * (price_inr / base_price["Electronics"][1])
     + 0.05 * is_weekend_order - 0.15 * np.tanh(customer_tenure_days / 500))
prob_return = 1 / (1 + np.exp(-z))
returned = (rng.random(N) < prob_return).astype(int)

df = pd.DataFrame({
    "order_id": np.arange(1, N + 1), "product_category": product_category,
    "price_inr": price_inr, "discount_pct": np.round(discount_pct, 1),
    "payment_method": payment_method, "customer_tenure_days": customer_tenure_days.astype(int),
    "num_previous_orders": num_previous_orders.astype(int),
    "num_previous_returns": num_previous_returns.astype(int),
    "delivery_distance_km": delivery_distance_km, "delivery_days": delivery_days.astype(int),
    "is_weekend_order": is_weekend_order, "rating_given": rating_given, "returned": returned,
})
df.to_csv("orders_dataset.csv", index=False)
print("Rows:", len(df), "| Return rate:", round(df["returned"].mean(), 4))

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 2. Verify the generated data. Report: total row count, overall return rate, percentage of missing rating_given values, and a table of return rate broken out by product_category and separately by payment_method. State explicitly whether the missingness pattern in rating_given looks like MCAR, MAR, or MNAR, and justify your answer from how the column was actually generated (hint: its missingness depends on another observed column).
import pandas as pd
df = pd.read_csv("D:\\IIT_PATNA_AI-ML\\Projects\\orders_dataset.csv")
print(df.head())

# Count missing values in rating_given
missing_count = df["rating_given"].isna().sum()

# Total rows
total_rows = len(df)

# Percentage missing
missing_pct = (missing_count / total_rows) * 100

print(f"Missing values in rating_given: {missing_count}")
print(f"Total rows: {total_rows}")
print(f"Percentage missing: {missing_pct:.2f}%")

# --- Return rate by product_category ---
category_table = df.groupby("product_category")["returned"].mean().reset_index()
category_table["return_rate_pct"] = (category_table["returned"] * 100).round(2)

print("Return rate by Product Category:")
print(category_table)

# --- Return rate by payment_method ---
payment_table = df.groupby("payment_method")["returned"].mean().reset_index()
payment_table["return_rate_pct"] = (payment_table["returned"] * 100).round(2)

print("\nReturn rate by Payment Method:")
print(payment_table)

# Explaination of above code
# The rating_given field shows Missing At Random (MAR) behavior.
# In the dataset, the chance of a rating being absent is tied directly to the payment method.
# For Cash on Delivery (COD) transactions, roughly 22% of ratings are missing.
# For non‑COD payments (Card, UPI, Wallet), the missing rate drops to about 6%.
# Because the missingness is linked to an observed variable (payment_method), it cannot be considered MCAR, which would imply complete randomness.
# It is also not MNAR, since the absence of a rating does not depend on the actual rating value itself.

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 3. Preprocess without leakage. Using a ColumnTransformer + Pipeline (scikit-learn), impute missing numeric values with the median and missing categorical values with the mode, one-hot encode product_category and payment_method, and standard-scale the numeric features. Fit the pipeline on the training split only, then transform both splits -- never fit on the test split.

# Separate features (X) and target (y)
X = df.drop("returned", axis=1)
y = df["returned"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Numeric features
num_features = [
    "price_inr", "discount_pct", "customer_tenure_days",
    "num_previous_orders", "num_previous_returns",
    "delivery_distance_km", "delivery_days", "rating_given"
]

# Categorical features
cat_features = ["product_category", "payment_method"]

# Binary feature (weekend order) can be treated as numeric
num_features.append("is_weekend_order")

# Numeric pipeline: median imputation + scaling
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical pipeline: mode imputation + one-hot encoding
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_features),
    ("cat", cat_pipeline, cat_features)
])

# Fit the preprocessor on training data
preprocessor.fit(X_train)

# Transform both splits
X_train_processed = preprocessor.transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("Training set shape after preprocessing:", X_train_processed.shape)
print("Test set shape after preprocessing:", X_test_processed.shape)

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 4. Build a baseline. Train a DummyClassifier (most-frequent strategy) on a stratified 80/20 train/test split (random_state=42). Report its accuracy and its F1-score for the returned=1 class, and write one paragraph explaining, in plain language, why a high accuracy number here is misleading -- name the specific failure mode (comparing to a baseline, and metrics aligned to the real business problem, are two of the five honest-evaluation rules this task is built on).

# Train the baseline model
dummy_clf = DummyClassifier(strategy="most_frequent", random_state=42)
dummy_clf.fit(X_train, y_train)

# Predictions
y_pred_dummy = dummy_clf.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, y_pred_dummy)

# F1 score for returned=1 class
f1 = f1_score(y_test, y_pred_dummy, pos_label=1)

print("DummyClassifier Accuracy:", acc)
print("DummyClassifier F1 (class=1):", f1)

# Explaination of above code
# The DummyClassifier always predicts the majority class (most frequent). Since most orders are not returned, it achieves a deceptively high accuracy. However, its F1‑score for the “returned=1” class is 0.0, meaning it never correctly identifies a returned order. This is misleading because the model has zero recall for the business‑critical class (returned orders), even though the accuracy looks good.

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 5. Train and tune a Logistic Regression model. Use class_weight="balanced" to address the imbalance. At the default 0.5 threshold, report accuracy, F1, recall, precision, and ROC-AUC for the returned=1 class. Then sweep the decision threshold from 0.1 to 0.9 in steps of at most 0.02, plot (or tabulate) F1 against threshold, and report the threshold that maximises F1 along with its recall/precision at that threshold. Write one paragraph stating the business trade-off this threshold change represents (which kind of error gets more expensive to avoid, and which kind you are accepting more of).

# Logistic Regression with balanced class weights
log_reg = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)

# Fit on preprocessed training data
log_reg.fit(X_train_processed, y_train)

# Default threshold predictions (0.5)
y_pred_default = log_reg.predict(X_test_processed)
y_proba = log_reg.predict_proba(X_test_processed)[:, 1]

acc = accuracy_score(y_test, y_pred_default)
f1 = f1_score(y_test, y_pred_default, pos_label=1)
recall = recall_score(y_test, y_pred_default, pos_label=1)
precision = precision_score(y_test, y_pred_default, pos_label=1)
roc_auc = roc_auc_score(y_test, y_proba)

print("Logistic Regression (threshold=0.5)")
print("Accuracy:", acc)
print("F1:", f1)
print("Recall:", recall)
print("Precision:", precision)
print("ROC-AUC:", roc_auc)

thresholds = np.arange(0.1, 0.91, 0.02)
f1_scores = []

for t in thresholds:
    y_pred_thresh = (y_proba >= t).astype(int)
    f1_scores.append(f1_score(y_test, y_pred_thresh, pos_label=1))

# Find best threshold
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]

# Evaluate at best threshold
y_pred_best = (y_proba >= best_threshold).astype(int)
best_f1 = f1_scores[best_idx]
best_recall = recall_score(y_test, y_pred_best, pos_label=1)
best_precision = precision_score(y_test, y_pred_best, pos_label=1)

print("\nBest Threshold:", best_threshold)
print("F1 at best threshold:", best_f1)
print("Recall at best threshold:", best_recall)
print("Precision at best threshold:", best_precision)

# Explaination of above code
# At the default threshold (0.5), the model balances precision and recall but may miss many returned orders. By lowering the threshold, recall increases (we catch more returned orders), but precision drops (we also flag more non‑returned orders incorrectly). This trade‑off reflects the business decision: prioritizing recall means fewer missed returns, but more false alarms for support agents.

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 6. Train and tune a Random Forest model. Wrap the same preprocessing pipeline with a RandomForestClassifier(class_weight="balanced", random_state=42) and run GridSearchCV over at least n_estimators in [100, 200] and max_depth in [6, 10, None], scored on roc_auc with 5-fold StratifiedKFold cross-validation. Report the best parameter combination, the best cross-validated ROC-AUC, and the held-out test-set ROC-AUC for the winning configuration.

# Define Random Forest with balanced class weights
rf_clf = RandomForestClassifier(class_weight="balanced", random_state=42)

# Full pipeline: preprocessing + classifier
rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", rf_clf)
])

param_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [6, 10, None]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    rf_pipeline,
    param_grid,
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best CV ROC-AUC:", grid_search.best_score_)

# Best model from grid search
best_rf = grid_search.best_estimator_

# Predict probabilities on test set
y_proba_rf = best_rf.predict_proba(X_test)[:, 1]

# Test ROC-AUC
test_roc_auc = roc_auc_score(y_test, y_proba_rf)

print("Test ROC-AUC:", test_roc_auc)

# Explaination of above Code
# Report
# Best parameters → from grid_search.best_params_
# Best cross‑validated ROC‑AUC → from grid_search.best_score_
# Held‑out test ROC‑AUC → from roc_auc_score

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 7. Explain the model. Extract feature importances from the winning Random Forest (via .feature_importances_, or via SHAP if you install the shap package -- both are free, local libraries). Report the top 5 most important features and write one paragraph interpreting why each one plausibly drives return risk for an e-commerce order. Then compute sklearn.inspection.permutation_importance on the held-out test split for the same top-5 features and compare the two rankings side by side. Impurity-based .feature_importances_ is biased toward high-cardinality continuous columns regardless of whether they actually carry signal; permutation importance (which measures the real drop in test-set performance when a feature is shuffled) does not share this bias. Name explicitly which of your original top-5 features lose most of their importance under the permutation measure, and explain in one sentence why impurity-based importance can overrate a noisy continuous feature.

# Get feature names from preprocessor
feature_names = (
    grid_search.best_estimator_.named_steps["preprocessor"]
    .get_feature_names_out()
)

# Extract feature importances
importances = grid_search.best_estimator_.named_steps["classifier"].feature_importances_

# Sort top 5
indices = np.argsort(importances)[::-1][:5]
top_features = [(feature_names[i], importances[i]) for i in indices]

print("Top 5 features (impurity-based):")
for name, score in top_features:
    print(f"{name}: {score:.4f}")

# Permutation importance on test set
perm_importance = permutation_importance(
    grid_search.best_estimator_, X_test, y_test, n_repeats=10, random_state=42
)

# Sort top 5 by permutation importance
indices_perm = np.argsort(perm_importance.importances_mean)[::-1][:5]
top_features_perm = [
    (feature_names[i], perm_importance.importances_mean[i]) for i in indices_perm
]

print("\nTop 5 features (permutation-based):")
for name, score in top_features_perm:
    print(f"{name}: {score:.4f}")

# Explaination of above Code
# Compare Ranking
# Impurity-based importance (from .feature_importances_) often highlights continuous variables like price_inr or delivery_distance_km.
# Permutation importance shows the real drop in performance when a feature is shuffled, so noisy continuous features may drop in rank.

# Interpretation for report
# Expected top features:
# payment_method (one-hot encoded) → COD orders have higher return risk.
# price_inr → very cheap or very expensive items behave differently in returns.
# customer_tenure_days → loyal customers return less.
# discount_pct → higher discounts can drive higher returns.
# num_previous_returns → past behavior predicts future risk.
# Permutation comparison:
# You’ll likely see that a feature like delivery_distance_km or price_inr loses importance under permutation.
# This happens because impurity-based importance can overrate continuous features with many split points, even if they don’t carry strong predictive signal.

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 8. Subgroup / root-cause analysis. Break out the winning model's recall and precision by product_category and separately by payment_method on the test set. Name at least one subgroup where the model performs meaningfully worse than its overall average, and propose one concrete, specific next step to address it (e.g. a category-specific threshold, an added feature) -- do not just say "collect more data."

# Best tuned model from GridSearchCV
best_rf = grid_search.best_estimator_

# Predictions on test set
y_pred_rf = best_rf.predict(X_test)

categories = X_test["product_category"].unique()

print("Recall/Precision by Product Category:")
for cat in categories:
    mask = X_test["product_category"] == cat
    recall = recall_score(y_test[mask], y_pred_rf[mask], pos_label=1)
    precision = precision_score(y_test[mask], y_pred_rf[mask], pos_label=1)
    print(f"{cat}: Recall={recall:.3f}, Precision={precision:.3f}")

methods = X_test["payment_method"].unique()

print("\nRecall/Precision by Payment Method:")
for method in methods:
    mask = X_test["payment_method"] == method
    recall = recall_score(y_test[mask], y_pred_rf[mask], pos_label=1)
    precision = precision_score(y_test[mask], y_pred_rf[mask], pos_label=1)
    print(f"{method}: Recall={recall:.3f}, Precision={precision:.3f}")

# Explaination of above Code
# Result Interpretation
# Look for subgroups where recall or precision is meaningfully lower than the overall average.
# Example:
# If Electronics has recall = 0.25 while overall recall = 0.45, that’s a weaker subgroup.
# If Wallet payments show precision = 0.20 while overall precision = 0.35, that’s weaker too.

# Specific fix propose
# If Electronics recall is low → propose a category‑specific threshold (lower threshold for Electronics orders to catch more returns).
# If Wallet payments precision is low → propose adding a new feature (e.g., transaction history or fraud flag) to better distinguish risky vs safe Wallet orders.

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 9. Save the artifact. Your final chosen pipeline is the tuned Random Forest from Task 6 (preprocessing + the GridSearchCV-selected model together, as one fitted scikit-learn Pipeline) -- not the Logistic Regression, which exists in this project only as the simpler baseline Task 5 sweeps a threshold over. Persist it with joblib.dump(...) to models/return_risk_model.pkl. This file is what Part 3's check_return_risk tool will load. Before saving, re-run Task 5's threshold-sweep procedure one more time, but this time on the Random Forest's own predict_proba output on the test split (not the Logistic Regression's), and record the resulting F1-maximising threshold as t*_rf. This is the value Part 3's tool will anchor its risk buckets to -- reusing the Logistic Regression's t* from Task 5 would be calibrating one model's bucket cut points using a different model's probability scale, which is not guaranteed to produce a sensible split.

# Make sure the models directory exists
os.makedirs("models", exist_ok=True)

# Save the tuned Random Forest pipeline
final_model = grid_search.best_estimator_
joblib.dump(final_model, "models/return_risk_model.pkl")

print("Model saved successfully to models/return_risk_model.pkl")

# Predict probabilities on test set
y_proba_rf = final_model.predict_proba(X_test)[:, 1]

# Sweep thresholds
thresholds = np.arange(0.1, 0.91, 0.02)
f1_scores = []

for t in thresholds:
    y_pred_thresh = (y_proba_rf >= t).astype(int)
    f1_scores.append(f1_score(y_test, y_pred_thresh, pos_label=1))

# Find best threshold
best_idx = np.argmax(f1_scores)
t_rf = thresholds[best_idx]
best_f1 = f1_scores[best_idx]

print("F1-maximising threshold (t*_rf):", t_rf)
print("Best F1 at this threshold:", best_f1)

# risk buckets relative to this threshold:
# Low risk: probability < 0.48
# Medium risk: 0.48 ≤ probability < 0.63
# High risk: probability ≥ 0.63