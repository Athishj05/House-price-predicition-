import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv(r"E:\project.athish\dataset files\house_prices.csv")
print("HOUSE PRICE PREDICTION")
#1. dataset info
df=df.head()
print("Dataset Shape:")
print(df.shape)

print("Column Names:")
print(df.columns.tolist())

print("Missing Values:")
print(df.isnull().sum())

# 2. CLEAN COLUMN NAMES
df.columns = df.columns.str.strip()


# 3. SELECT PRICE COLUMN


price_column = "Price (in rupees)"

if price_column not in df.columns:
    print("ERROR: Price column was not found.")
    print("Available columns are:")
    print(df.columns.tolist())
    exit()

print("Price column found:", price_column)

# 4. CONVERT PRICE TO NUMERIC

df[price_column] = (df[price_column].astype(str).str.replace(",", "", regex=False).str.replace("₹", "", regex=False).str.strip())
df[price_column] = pd.to_numeric(df[price_column],errors="coerce")

# 5. REMOVE ROWS WHERE PRICE IS MISSING


df = df.dropna(subset=[price_column])
print("Rows after removing missing prices:")
print(len(df))

# 6. SELECT FEATURES

features = ["Carpet Area","Bathroom","Balcony"]


# Check which features actually exist
available_features = []
for column in features:
    if column in df.columns:
        available_features.append(column)


print("Available Features:")
print(available_features)

# 7. CONVERT FEATURES TO NUMERIC

for column in available_features:

    df[column] = (df[column].astype(str).str.replace(",", "", regex=False).str.extract(r"(\d+\.?\d*)")[0])

    df[column] = pd.to_numeric(df[column],errors="coerce")

# 8. CREATE X AND Y

X = df[available_features].copy()
y = df[price_column].copy()

# 9. FILL MISSING FEATURE VALUES

X = X.fillna(X.median())
print("Features used:")
print(X.columns.tolist())
print("Feature Data:")
print(X.head())

# 10. TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=42)

print("Training Data:")
print(X_train.shape)
print("Testing Data:")
print(X_test.shape)

# 11. CREATE MODEL

model = LinearRegression()
# 12. TRAIN MODEL
model.fit(X_train, y_train)
print("Model Training Completed!")

# 13. MAKE PREDICTIONS

y_pred = model.predict(X_test)

# 14. MODEL EVALUATION

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("MODEL PERFORMANCE")
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R2 Score:", r2)

# 15. ACTUAL VS PREDICTED

result = pd.DataFrame({ "Actual Price": y_test.values,"Predicted Price": y_pred})

print("Actual vs Predicted:")
print(result.head(10))

# 16. VISUALIZATION

plt.figure(figsize=(8, 6))
plt.scatter(y_test,y_pred,alpha=0.5)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.show()

# 17. FEATURE IMPORTANCE / COEFFICIENTS

print("FEATURE COEFFICIENTS")
for feature, coefficient in zip(available_features,model.coef_):
    print(feature, ":", coefficient)

# 18. PREDICT PRICE FOR A NEW HOUSE

print("NEW HOUSE PRICE PREDICTION")


new_house = pd.DataFrame({
    "Carpet Area": [1000],
    "Bathroom": [2],
    "Balcony": [1]})

# Keep only features used by model
new_house = new_house[available_features]
predicted_price = model.predict(new_house)
print("Predicted House Price:", predicted_price[0])


# 19. SAVE PREDICTIONS
result.to_csv("house_price_predictions.csv",index=False)
print("Prediction results saved as:")
print("house_price_predictions.csv")

# 20. SAVE CLEANED DATA

cleaned_data = df[available_features + [price_column]]
cleaned_data.to_csv("house_prices_cleaned.csv",index=False)
print("Cleaned dataset saved as:")
print("house_prices_cleaned.csv")
