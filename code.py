import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Sample data for house prices
data = {
    'Location': ['Suburban', 'Urban', 'Rural', 'Suburban', 'Urban', 'Rural', 'Suburban', 'Urban', 'Rural', 'Suburban'],
    'Size (sqft)': [2000, 1500, 2500, 1800, 1600, 3000, 2100, 1400, 2400, 1700],
    'Bedrooms': [3, 2, 4, 3, 2, 4, 3, 2, 3, 3],
    'Bathrooms': [2, 1, 3, 2, 1, 3, 2, 1, 2, 2],
    'Price ($)': [250000, 300000, 220000, 230000, 320000, 210000, 260000, 330000, 225000, 240000]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Features (X) and target (y)
X = df[['Location', 'Size (sqft)', 'Bedrooms', 'Bathrooms']]
y = df['Price ($)']

# Preprocessing for numeric and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['Size (sqft)', 'Bedrooms', 'Bathrooms']), 
        ('cat', OneHotEncoder(), ['Location'])
    ]
)

# Create a pipeline with preprocessing and the linear regression model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor), 
    ('model', LinearRegression())
])

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the pipeline on the training data
pipeline.fit(X_train, y_train)

# Make predictions on the test set
y_pred = pipeline.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the evaluation metrics
print(f'Mean Absolute Error (MAE): {mae}')
print(f'Mean Squared Error (MSE): {mse}')
print(f'R-squared (R2): {r2}')

# Predict the price for a new house
new_data = pd.DataFrame({
    'Location': ['Urban'],
    'Size (sqft)': [1800],
    'Bedrooms': [3],
    'Bathrooms': [2]
})

predicted_price = pipeline.predict(new_data)

# Print the predicted price for the new data
print(f'Predicted Price for new data: ${predicted_price[0]:,.2f}')