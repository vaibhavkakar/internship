
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
url = 'https://github.com/FlipRoboTechnologies/ML-Datasets/blob/main/Grades/Grades.csv?raw=true'
grades_df = pd.read_csv(url)

# Display the first few rows of the dataset
print(grades_df.head())

# Get a summary of the dataset
print(grades_df.info())

# Check for missing values
print(grades_df.isnull().sum())

# Statistical summary of the dataset
print(grades_df.describe())

# Split the data into features and target
X = grades_df.drop(columns=['CGPA'])
y = grades_df['CGPA']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R^2 Score:", r2_score(y_test, y_pred))
