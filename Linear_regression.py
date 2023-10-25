import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Load the modified dataset
data = pd.DataFrame({
    'Accident_ID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Weather_Condition': ['Clear', 'Rain', 'Fog', 'Snow', 'Clear', 'Rain', 'Fog', 'Snow', 'Clear', 'Rain'],
    'Road_Type': ['Highway', 'City', 'Rural', 'City', 'Rural', 'City', 'Highway', 'City', 'Rural', 'City'],
    'Speed_Limit': [70, 30, 50, 40, 60, 30, 70, 40, 60, 30],
    'Time_of_Day': ['Morning', 'Afternoon', 'Evening', 'Night', 'Morning', 'Afternoon', 'Evening', 'Night', 'Morning', 'Afternoon'],
    'Alcohol_Involved': ['No', 'Yes', 'No', 'No', 'No', 'No', 'Yes', 'Yes', 'No', 'No'],
    'Fatality': ['No', 'No', 'Yes', 'No', 'No', 'No', 'Yes', 'Yes', 'No', 'No'],
    'Injury_Severity': ['Minor', 'Major', 'Fatal', 'Minor', 'Minor', 'Minor', 'Major', 'Fatal', 'Minor', 'Minor']
})

# Encode categorical variables using one-hot encoding
data = pd.get_dummies(data, columns=['Weather_Condition', 'Road_Type', 'Time_of_Day', 'Alcohol_Involved', 'Fatality'])

# Encode the target variable (Injury_Severity) using label encoding
label_encoder = LabelEncoder()
data['Injury_Severity'] = label_encoder.fit_transform(data['Injury_Severity'])

# Define the dependent and independent variables
X = data.drop(['Accident_ID', 'Injury_Severity'], axis=1)
y = data['Injury_Severity']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit a decision tree regression model
model = DecisionTreeRegressor()
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')
# Save the model for future use
joblib.dump(model, 'accident_severity_model.pkl')