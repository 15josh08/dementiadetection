import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('oasis_longitudinal.csv')

# Filter for the first visit data
df = df.loc[df['Visit'] == 1]
df = df.reset_index(drop=True)

# Convert categorical variables to numerical
df['M/F'] = df['M/F'].replace({'F': 0, 'M': 1})
df['Group'] = df['Group'].replace({'Converted': 'Demented'})
df['Group'] = df['Group'].replace({'Demented': 1, 'Nondemented': 0})

# Drop unnecessary columns
df = df.drop(['MRI ID', 'Visit', 'Hand'], axis=1)

# Handle missing values
df['SES'].fillna(df.groupby('EDUC')['SES'].transform('median'), inplace=True)
df = df.dropna()

# Prepare features and target
Y = df['Group'].values
X = df[['M/F','Age', 'EDUC', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF']]

# Split the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)

# Feature scaling
scaler = MinMaxScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [5, 10, 15, None]
}

# Initialize the RandomForestClassifier
rf_model = RandomForestClassifier(random_state=0)

# Initialize GridSearchCV
grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='accuracy')

# Perform Grid Search to find the best parameters
grid_search.fit(X_train_scaled, Y_train)

# Get the best parameters and model
best_params = grid_search.best_params_
best_rf_model = grid_search.best_estimator_

# Train the model with the best parameters
best_rf_model.fit(X_train_scaled, Y_train)

# Predict on the test set
Y_pred = best_rf_model.predict(X_test_scaled)

# Calculate accuracy
accuracy = accuracy_score(Y_test, Y_pred)
print("Accuracy after hyperparameter tuning:", accuracy)

# Save the trained model to a pickle file
with open('random_forest_model.pkl', 'wb') as f:
    pickle.dump(best_rf_model, f)

# Save the scaler object to a pickle file
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
