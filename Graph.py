import matplotlib.pyplot as plt

# Create a bar plot to compare actual vs predicted house prices
plt.figure(figsize=(10, 6))
x_axis = range(len(y_test))

# Plotting actual vs predicted prices
plt.bar(x_axis, y_test, width=0.4, label='Actual Prices', align='center')
plt.bar(x_axis, y_pred, width=0.4, label='Predicted Prices', align='edge')

# Add labels and title
plt.xlabel('Sample Index')
plt.ylabel('Price ($)')
plt.title('Actual vs Predicted House Prices')
plt.legend()

# Show the plot
plt.show()
