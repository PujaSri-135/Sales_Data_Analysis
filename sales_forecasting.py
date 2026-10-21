
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# STEP 1: Load data
df = pd.read_csv(
    r"C:\Users\Teju\OneDrive\Desktop\Sales_Data_Analysis\data\retail_sales_dataset.csv"
)
df.columns = df.columns.str.strip()
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Total Amount"] = pd.to_numeric(
    df["Total Amount"], errors="coerce"
)

df = df.dropna(subset=["Date", "Total Amount"])

# STEP 2: Calculate daily sales
daily = (
    df.groupby("Date")["Total Amount"]
    .sum()
    .sort_index()
)

# Use a continuous daily calendar.
# Only do this if missing dates truly represent zero recorded sales.
daily = daily.asfreq("D", fill_value=0)
daily.name = "Sales"

print("Daily sales data:")
print(daily.head())

# STEP 3: Create features from past sales
data = pd.DataFrame({"Sales": daily})

data["Yesterday"] = data["Sales"].shift(1)
data["Sales_7_Days_Ago"] = data["Sales"].shift(7)

data["Average_Last_7_Days"] = (
    data["Sales"].shift(1).rolling(7).mean()
)

data["DayOfWeek"] = data.index.dayofweek
data["Month"] = data.index.month

# Remove rows without enough historical information
data = data.dropna()

features = [
    "Yesterday",
    "Sales_7_Days_Ago",
    "Average_Last_7_Days",
    "DayOfWeek",
    "Month"
]

X = data[features]
y = data["Sales"]

# STEP 4: Split by date (no random shuffling)
split = int(len(data) * 0.80)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

if len(X_train) == 0 or len(X_test) == 0:
    raise ValueError("Not enough data to train and test the model.")

# STEP 5: Train the model
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

print("\nModel training completed!")

# STEP 6: Predict sales for the test period
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print("\nModel Evaluation")
print("----------------")
print("Mean Absolute Error:", round(mae, 2))
print("Root Mean Squared Error:", round(rmse, 2))

# STEP 7: Plot actual vs predicted sales
plt.figure(figsize=(12, 5))

plt.plot(
    y_test.index,
    y_test.values,
    label="Actual Sales"
)

plt.plot(
    y_test.index,
    predictions,
    label="Predicted Sales"
)

plt.title("Actual vs Predicted Daily Sales")
plt.xlabel("Date")
plt.ylabel("Sales Amount")
plt.legend()
plt.tight_layout()
plt.savefig("sales_prediction_evaluation.png", dpi=300)
plt.show()

# STEP 8: Forecast the next 7 days
history = daily.copy()
future_results = []

for _ in range(7):
    next_date = history.index[-1] + pd.Timedelta(days=1)

    row = pd.DataFrame([{
        "Yesterday": history.iloc[-1],
        "Sales_7_Days_Ago": history.iloc[-7],
        "Average_Last_7_Days": history.iloc[-7:].mean(),
        "DayOfWeek": next_date.dayofweek,
        "Month": next_date.month
    }], index=[next_date])

    predicted_sales = max(0, model.predict(row[features])[0])

    future_results.append({
        "Date": next_date,
        "Predicted Sales": round(predicted_sales, 2)
    })

    # Use the prediction as input for the following forecast day
    history.loc[next_date] = predicted_sales

forecast = pd.DataFrame(future_results)

print("\nNext 7 Days Sales Forecast")
print(forecast.to_string(index=False))

# STEP 9: Save forecast results
forecast.to_csv("future_sales_forecast.csv", index=False)

print("\nSaved: future_sales_forecast.csv")
print("Saved: sales_prediction_evaluation.png")
