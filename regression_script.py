import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load the Excel dataset
data = pd.read_excel('aqi_dataset_processed.xlsx')

# 2. Define independent variable (X) and dependent variable (y)
# X must be a 2D structure, so we use double brackets [['PM2.5']]
X = data[['PM2.5']]
y = data['AQI']

# 3. Create and train the Linear Regression model
model = LinearRegression()
model.fit(X, y)

# 4. Extract Intercept (b0) and Slope (b1)
b0 = model.intercept_
b1 = model.coef_[0]

print('--- MODEL PARAMETERS ---')
print(f'Intercept (b0): {b0:.2f}')
print(f'Slope (b1): {b1:.2f}')
print(f'Equation: y_hat = {b0:.2f} + {b1:.2f} * x\n')

# 5. Make predictions (y_hat)
y_pred = model.predict(X)

# 6. Evaluate model performance using your metrics
mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = mse**0.5
r2 = r2_score(y, y_pred)

print('--- EVALUATION METRICS ---')
print(f'Mean Absolute Error (MAE): {mae:.2f}')
print(f'Mean Squared Error (MSE): {mse:.2f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')
print(f'R-squared (R2): {r2:.4f}\n')

# 7. Plot the data points and the regression line
plt.scatter(X, y, color='blue', alpha=0.5, label='Actual Data')
plt.plot(
    X, y_pred, color='red', linewidth=2, label='Best-Fit Regression Line'
)
plt.xlabel('PM2.5')
plt.ylabel('AQI')
plt.title('Linear Regression: PM2.5 vs AQI')
plt.legend()
plt.grid(True)
plt.show()
