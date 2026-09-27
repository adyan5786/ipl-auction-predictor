"""Prepare the IPL auction data for a later machine learning step."""

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor


# Step 1: Load the merged auction and player statistics data.
df = pd.read_csv("data/ipl_merged.csv")

# Step 2: Inspect the data and check how many values are missing.
print("DataFrame information:")
df.info()
print("\nMissing values in each column:")
print(df.isnull().sum())

# Step 3: Fill missing numeric values with the mean of their column.
numeric_columns = df.select_dtypes(include=[np.number]).columns
for column in numeric_columns:
	df[column] = df[column].fillna(df[column].mean())

# Step 4: Fill missing categorical values with the most common value.
categorical_columns = df.select_dtypes(include=["object", "category"]).columns
for column in categorical_columns:
	if df[column].isnull().any():
		df[column] = df[column].fillna(df[column].mode()[0])

# Step 5: Inspect auction-year counts and average prices.
print("\nNumber of auction rows per Year:")
print(df["Year"].value_counts().sort_index())
print("\nAverage Amount by Year:")
print(df.groupby("Year")["Amount"].mean())

# Step 6: Discussion-only year normalization.
# This comparison is kept for the report but is not used by the final model.
year_avg_amount = df.groupby("Year")["Amount"].transform("mean")
overall_avg_amount = df["Amount"].mean()
df["Year_Normalized_Amount"] = (
	df["Amount"] / year_avg_amount
) * overall_avg_amount
print("\nOriginal and year-normalized amounts (first 10 rows):")
print(df[["Amount", "Year_Normalized_Amount"]].head(10))
print(f"\nTraining Year range: {df['Year'].min()} to {df['Year'].max()}")

# Step 7: Convert categorical text values into numbers.
# Each encoder is saved so app.py can reuse it later for input decoding.
label_encoders = {}
for column in categorical_columns:
	encoder = LabelEncoder()
	df[column] = encoder.fit_transform(df[column].astype(str))
	label_encoders[column] = encoder

# Step 8: Separate the target from the features.
columns_to_exclude = [
	"Player",
	"Unnamed: 0",
	"player",
	"auction_player_name",
	"matched_stats_player_name",
	"Amount",
	"Year_Normalized_Amount",
]
X = df.drop(columns=columns_to_exclude)
y = df["Amount"]

# Step 9: Split the data for later model training and evaluation.
X_train, X_test, y_train, y_test = train_test_split(
	X,
	y,
	test_size=0.2,
	random_state=42,
)

# Step 10: Scale numeric features using training data only.
# Label-encoded categorical columns stay unscaled because their numeric labels
# are categories, not measurements with meaningful distances.
categorical_feature_columns = ["Role", "Team", "Player Origin"]
numeric_feature_columns = [
	column for column in X.columns if column not in categorical_feature_columns
]
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[numeric_feature_columns] = scaler.fit_transform(
	X_train[numeric_feature_columns]
)
X_test_scaled[numeric_feature_columns] = scaler.transform(
	X_test[numeric_feature_columns]
)
joblib.dump(scaler, "model/scaler.pkl")

print("\nPreprocessing complete.")
print(f"Training features shape: {X_train.shape}")
print(f"Testing features shape: {X_test.shape}")

# Step 11: Train a simple linear regression model on scaled features.
linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)
linear_predictions = linear_model.predict(X_test_scaled)
linear_mae = mean_absolute_error(y_test, linear_predictions)
linear_r2 = r2_score(y_test, linear_predictions)
print("\nLinear Regression Results:")
print(f"Mean Absolute Error (MAE): {linear_mae:.2f}")
print(f"R2 Score: {linear_r2:.4f}")

# Step 12: Train a shallow decision tree for comparison on scaled features.
tree_model = DecisionTreeRegressor(max_depth=5, random_state=42)
tree_model.fit(X_train_scaled, y_train)
tree_predictions = tree_model.predict(X_test_scaled)
tree_mae = mean_absolute_error(y_test, tree_predictions)
tree_r2 = r2_score(y_test, tree_predictions)
print("\nDecision Tree Results:")
print(f"Mean Absolute Error (MAE): {tree_mae:.2f}")
print(f"R2 Score: {tree_r2:.4f}")

# Step 13: Diagnose Year relationships and Linear Regression coefficients.
print("\nCorrelation of Year with numeric training features:")
print(df[numeric_feature_columns].corr()["Year"].sort_values())

coefficient_series = pd.Series(linear_model.coef_, index=X.columns)
sorted_coefficients = coefficient_series.reindex(
	coefficient_series.abs().sort_values(ascending=False).index
)
print("\nLinear Regression coefficients sorted by absolute value:")
print(sorted_coefficients)
print(f"\nLinear Regression coefficient for Year: {coefficient_series['Year']:.6f}")

# Step 14: Select the model with the higher R2 score.
if linear_r2 >= tree_r2:
	best_model = linear_model
	best_model_name = "Linear Regression"
	best_r2 = linear_r2
else:
	best_model = tree_model
	best_model_name = "Decision Tree"
	best_r2 = tree_r2

# Step 15: Save the selected model and preprocessing information for app.py.
joblib.dump(best_model, "model/model.pkl")
joblib.dump(label_encoders, "model/label_encoders.pkl")
joblib.dump(X.columns.tolist(), "model/feature_columns.pkl")

print(f"\nSelected model: {best_model_name}")
print(f"Selected model R2 score: {best_r2:.4f}")
print("Saved model, scaler, encoders, and feature columns to the model/ folder.")