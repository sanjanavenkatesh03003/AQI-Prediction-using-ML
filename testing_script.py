import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# 1. Load your processed Excel dataset
data = pd.read_excel('aqi_dataset_processed.xlsx')

# 2. Define your input features (X) and target variable (y)
X = data[['PM2.5', 'Temp', 'Humidity', 'CO', 'O3']]
y = data['AQI']

# 3. Split the dataset: 80% for training the model, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Make predictions using the unseen test data
y_pred = model.predict(X_test)

# 6. Combine Actual AQI and Predicted AQI into a DataFrame
comparison_df = pd.DataFrame(
    {'Actual_AQI': y_test.values, 'Predicted_AQI': y_pred}
)

# 7. Save the results to a CSV file
output_filename = 'linear_regression_test_predictions.csv'
comparison_df.to_csv(output_filename, index=False)

print(f'Success! Predictions saved to {output_filename}')
print(comparison_df.head(10))  # Displays the first 10 rows in your terminal
