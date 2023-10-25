import joblib
import pandas as pd

# Load the saved model
loaded_model = joblib.load('accident_severity_model.pkl')

# Define hypothetical independent variables for prediction
new_data = pd.DataFrame({
    'Weather_Condition_Clear': [1],  # Replace with the actual values for independent variables
    'Road_Type_City': [1],
    'Speed_Limit': [40.0],
    'Time_of_Day_Evening': [1],
    'Alcohol_Involved_No': [1],
    'Fatality_No': [1]
})

# Make predictions directly
predicted_severity = loaded_model.predict(new_data)
print(f'Predicted Accident Severity: {predicted_severity[0]}')
