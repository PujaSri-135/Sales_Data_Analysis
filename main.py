import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# SALES DATA ANALYSIS PROJECT
# Dataset: retail_sales_dataset.csv
# ============================================================


# ------------------------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------------------------

df = pd.read_csv("data/retail_sales_dataset.csv")

print("=" * 60)
print("          SALES DATA ANALYSIS PROJECT")
print("=" * 60)

print("\nDataset loaded successfully!")


# ------------------------------------------------------------
# 2. BASIC INFORMATION ABOUT DATASET
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print("\nNumber of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 5 records:")
print(df.head())


# ------------------------------------------------------------
# 3. CHECK DATA TYPES
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("DATA TYPES")
print("=" * 60)

print(df.dtypes)


# ------------------------------------------------------------
# 4. CHECK MISSING VALUES
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

missing_values = df.isnull().sum()

print(missing_values)


# ------------------------------------------------------------
# 5. REMOVE DUPLICATE RECORDS
# ------------------------------------------------------------

duplicate_count = df.duplicated().sum()

print("\nNumber of duplicate records:", duplicate_count)

if duplicate_count > 0:
    df = df.drop_duplicates()
    print("Duplicate records removed.")
else:
    print("No duplicate records found.")


# ------------------------------------------------------------
# 6. CONVERT DATE COLUMN
# ------------------------------------------------------------

df["Date"] = pd.to_datetime(df["Date"])

print("\nDate column converted successfully.")


# ------------------------------------------------------------
# 7. BASIC STATISTICS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("BASIC STATISTICS")
print("=" * 60)

print(df.describe())


# ------------------------------------------------------------
# 8. TOTAL SALES
# ------------------------------------------------------------

total_sales = df["Total Amount"].sum()

print("\nTotal Sales: ₹", total_sales)


# ------------------------------------------------------------
# 9. TOTAL QUANTITY SOLD
# ------------------------------------------------------------

total_quantity = df["Quantity"].sum()

print("Total Quantity Sold:", total_quantity)


# ------------------------------------------------------------
# 10. AVERAGE SALES PER TRANSACTION
# ------------------------------------------------------------

average_transaction = df["Total Amount"].mean()

print("Average Sales per Transaction: ₹", round(average_transaction, 2))


# ------------------------------------------------------------
# 11. TOTAL NUMBER OF CUSTOMERS
# ------------------------------------------------------------

total_customers = df["Customer ID"].nunique()

print("Total Unique Customers:", total_customers)


# ------------------------------------------------------------
# 12. SALES BY PRODUCT CATEGORY
# ------------------------------------------------------------

category_sales = (
    df.groupby("Product Category")["Total Amount"]
    .sum()
    .sort_values(ascending=False)
)

print("\n" + "=" * 60)
print("SALES BY PRODUCT CATEGORY")
print("=" * 60)

print(category_sales)


# ------------------------------------------------------------
# 13. QUANTITY SOLD BY CATEGORY
# ------------------------------------------------------------

category_quantity = (
    df.groupby("Product Category")["Quantity"]
    .sum()
    .sort_values(ascending=False)
)

print("\n" + "=" * 60)
print("QUANTITY SOLD BY CATEGORY")
print("=" * 60)

print(category_quantity)


# ------------------------------------------------------------
# 14. BEST SELLING CATEGORY
# ------------------------------------------------------------

best_category = category_sales.idxmax()
best_category_sales = category_sales.max()

print("\nBest Selling Category:", best_category)
print("Sales Generated: ₹", best_category_sales)


# ------------------------------------------------------------
# 15. GENDER-WISE SALES
# ------------------------------------------------------------

gender_sales = (
    df.groupby("Gender")["Total Amount"]
    .sum()
    .sort_values(ascending=False)
)

print("\n" + "=" * 60)
print("GENDER-WISE SALES")
print("=" * 60)

print(gender_sales)


# ------------------------------------------------------------
# 16. GENDER-WISE TRANSACTIONS
# ------------------------------------------------------------

gender_transactions = df["Gender"].value_counts()

print("\n" + "=" * 60)
print("NUMBER OF TRANSACTIONS BY GENDER")
print("=" * 60)

print(gender_transactions)


# ------------------------------------------------------------
# 17. AGE ANALYSIS
# ------------------------------------------------------------

average_age = df["Age"].mean()

print("\nAverage Customer Age:", round(average_age, 2))


youngest_customer = df["Age"].min()
oldest_customer = df["Age"].max()

print("Youngest Customer Age:", youngest_customer)
print("Oldest Customer Age:", oldest_customer)


# ------------------------------------------------------------
# 18. CREATE AGE GROUPS
# ------------------------------------------------------------

bins = [0, 18, 30, 40, 50, 60, 100]

labels = [
    "Below 18",
    "18-30",
    "31-40",
    "41-50",
    "51-60",
    "60+"
]

df["Age Group"] = pd.cut(
    df["Age"],
    bins=bins,
    labels=labels,
    include_lowest=True
)


# ------------------------------------------------------------
# 19. SALES BY AGE GROUP
# ------------------------------------------------------------

age_group_sales = (
    df.groupby(
        "Age Group",
        observed=True
    )["Total Amount"]
    .sum()
)

print("\n" + "=" * 60)
print("SALES BY AGE GROUP")
print("=" * 60)

print(age_group_sales)


# ------------------------------------------------------------
# 20. MONTHLY SALES
# ------------------------------------------------------------

df["Month"] = df["Date"].dt.month

df["Month Name"] = df["Date"].dt.strftime("%B")

monthly_sales = (
    df.groupby("Month")["Total Amount"]
    .sum()
    .sort_index()
)

print("\n" + "=" * 60)
print("MONTHLY SALES")
print("=" * 60)

print(monthly_sales)


# ------------------------------------------------------------
# 21. BEST SALES MONTH
# ------------------------------------------------------------

best_month_number = monthly_sales.idxmax()
best_month_sales = monthly_sales.max()

best_month_name = pd.to_datetime(
    str(best_month_number),
    format="%m"
).strftime("%B")

print("\nBest Sales Month:", best_month_name)
print("Sales in Best Month: ₹", best_month_sales)


# ------------------------------------------------------------
# 22. DAILY SALES
# ------------------------------------------------------------

daily_sales = (
    df.groupby("Date")["Total Amount"]
    .sum()
)

print("\nHighest Daily Sales: ₹", daily_sales.max())


# ------------------------------------------------------------
# 23. MOST POPULAR PRODUCT CATEGORY
# ------------------------------------------------------------

popular_category = category_quantity.idxmax()

print("Most Popular Category:", popular_category)


# ------------------------------------------------------------
# 24. PRICE ANALYSIS
# ------------------------------------------------------------

average_price = df["Price per Unit"].mean()

highest_price = df["Price per Unit"].max()
lowest_price = df["Price per Unit"].min()

print("\n" + "=" * 60)
print("PRICE ANALYSIS")
print("=" * 60)

print("Average Price per Unit: ₹", round(average_price, 2))
print("Highest Price per Unit: ₹", highest_price)
print("Lowest Price per Unit: ₹", lowest_price)


# ============================================================
# VISUALIZATION
# ============================================================


# ------------------------------------------------------------
# 25. CATEGORY SALES BAR CHART
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

category_sales.plot(kind="bar")

plt.title("Sales by Product Category")
plt.xlabel("Product Category")
plt.ylabel("Total Sales")

plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "category_sales.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()


# ------------------------------------------------------------
# 26. MONTHLY SALES LINE CHART
# ------------------------------------------------------------

plt.figure(figsize=(10, 5))

plt.plot(
    monthly_sales.index,
    monthly_sales.values,
    marker="o"
)

plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Total Sales")

plt.xticks(range(1, 13))

plt.grid(True)

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "monthly_sales.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ------------------------------------------------------------
# 27. GENDER-WISE SALES BAR CHART
# ------------------------------------------------------------

plt.figure(figsize=(7, 5))

gender_sales.plot(kind="bar")

plt.title("Sales by Gender")
plt.xlabel("Gender")
plt.ylabel("Total Sales")

plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "gender_sales.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ------------------------------------------------------------
# 28. SALES BY AGE GROUP
# ------------------------------------------------------------

plt.figure(figsize=(9, 5))

age_group_sales.plot(kind="bar")

plt.title("Sales by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Total Sales")

plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "age_group_sales.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ------------------------------------------------------------
# 29. CATEGORY SALES PIE CHART
# ------------------------------------------------------------

plt.figure(figsize=(7, 7))

plt.pie(
    category_sales.values,
    labels=category_sales.index,
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Sales Distribution by Product Category")

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "category_sales.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ------------------------------------------------------------
# 30. AGE DISTRIBUTION
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.hist(
    df["Age"],
    bins=10
)

plt.title("Customer Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "age_distribution.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()


# ------------------------------------------------------------
# 31. SALES DISTRIBUTION
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.hist(
    df["Total Amount"],
    bins=20
)

plt.title("Sales Amount Distribution")
plt.xlabel("Total Amount")
plt.ylabel("Number of Transactions")

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "sales_distribution.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()


# ------------------------------------------------------------
# 32. CORRELATION HEATMAP
# ------------------------------------------------------------

numeric_columns = [
    "Age",
    "Quantity",
    "Price per Unit",
    "Total Amount"
]

correlation = df[numeric_columns].corr()

plt.figure(figsize=(8, 6))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Between Numerical Variables")

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "correlation_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("                 FINAL SUMMARY")
print("=" * 60)

print("Total Sales: ₹", total_sales)

print("Total Quantity Sold:", total_quantity)

print("Average Transaction Value: ₹",
      round(average_transaction, 2))

print("Total Unique Customers:", total_customers)

print("Best Selling Category:", best_category)

print("Best Sales Month:", best_month_name)

print("Most Popular Category:", popular_category)

print("Average Customer Age:", round(average_age, 2))

print("\nAnalysis Completed Successfully!")

print("=" * 60)

from pathlib import Path
import json

import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from importlib import import_module
ColumnTransformer = import_module("sklearn.compose").ColumnTransformer

RandomForestClassifier = import_module("sklearn.ensemble").RandomForestClassifier
SimpleImputer = import_module("sklearn.impute").SimpleImputer
metrics = import_module("sklearn.metrics")
accuracy_score = metrics.accuracy_score
classification_report = metrics.classification_report
ConfusionMatrixDisplay = metrics.ConfusionMatrixDisplay
train_test_split = import_module("sklearn.model_selection").train_test_split
Pipeline = import_module("sklearn.pipeline").Pipeline
OneHotEncoder = import_module("sklearn.preprocessing").OneHotEncoder


# 1. Set project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "retail_sales_dataset.csv"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Place retail_sales_dataset.csv inside the data folder."
    )


# 2. Load and validate data
df = pd.read_csv(DATA_PATH)

required = [
    "Date", "Gender", "Age", "Product Category",
    "Quantity", "Price per Unit"
]

missing = [col for col in required if col not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
df["Price per Unit"] = pd.to_numeric(
    df["Price per Unit"], errors="coerce"
)

df = df.dropna(subset=["Date", "Product Category"])
df = df.dropna(subset=["Age", "Quantity", "Price per Unit"])

# 3. Create date-based features
df["Month"] = df["Date"].dt.month
df["DayOfWeek"] = df["Date"].dt.dayofweek

# Target: the category we want to predict
target = "Product Category"

# Exclude Total Amount and identifiers to avoid leakage
feature_columns = [
    "Gender", "Age", "Quantity", "Price per Unit",
    "Month", "DayOfWeek"
]

X = df[feature_columns]
y = df[target].astype(str)

if y.nunique() < 2:
    raise ValueError("At least two product categories are needed.")

if y.value_counts().min() < 2:
    raise ValueError(
        "Each category needs at least two records for a stratified split."
    )

# 4. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 5. Prepare numerical and categorical features
numeric_features = [
    "Age", "Quantity", "Price per Unit", "Month", "DayOfWeek"
]
categorical_features = ["Gender"]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# 6. Build and train the model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    ))
])

model.fit(X_train, y_train)

# 7. Evaluate predictions on unseen test data
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
report = classification_report(
    y_test, predictions, zero_division=0
)

print(f"Training records: {len(X_train)}")
print(f"Testing records: {len(X_test)}")
print(f"Accuracy: {accuracy:.2%}")
print("\nClassification report:\n")
print(report)

# 8. Save evaluation results
(OUTPUT_DIR / "model_report.txt").write_text(
    f"Product Category Classification\n"
    f"Training records: {len(X_train)}\n"
    f"Testing records: {len(X_test)}\n"
    f"Accuracy: {accuracy:.4f}\n\n"
    f"Classification report:\n{report}\n",
    encoding="utf-8"
)

results = X_test.copy()
results["Actual Category"] = y_test
results["Predicted Category"] = predictions
results.to_csv(
    OUTPUT_DIR / "model_predictions.csv", index=False
)

# 9. Save confusion matrix
fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay.from_predictions(
    y_test, predictions, ax=ax, xticks_rotation=45
)
ax.set_title("Product Category - Confusion Matrix")
fig.tight_layout()
fig.savefig(
    OUTPUT_DIR / "confusion_matrix.png",
    dpi=200,
    bbox_inches="tight"
)
plt.close(fig)

# 10. Save trained model
with (OUTPUT_DIR / "product_category_model.joblib").open("wb") as model_file:
    pickle.dump(model, model_file)

# Save a small metadata file
(OUTPUT_DIR / "model_metadata.json").write_text(
    json.dumps({
        "target": target,
        "features": feature_columns,
        "accuracy": round(float(accuracy), 4),
        "train_size": len(X_train),
        "test_size": len(X_test)
    }, indent=2),
    encoding="utf-8"
)

print(f"\nAll model files saved in:\n{OUTPUT_DIR.resolve()}")
print("Modelling completed successfully.")
