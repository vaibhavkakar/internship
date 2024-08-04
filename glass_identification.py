
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

# Load the dataset
url = 'https://github.com/FlipRoboTechnologies/ML-Datasets/blob/main/Glass%20Identification/Glass%20Identification.csv?raw=true'
glass_df = pd.read_csv(url)

# Display the first few rows of the dataset
print(glass_df.head())

# Get a summary of the dataset
print(glass_df.info())

# Check for missing values
print(glass_df.isnull().sum())

# Statistical summary of the dataset
print(glass_df.describe())

# Normalize the feature columns
scaler = StandardScaler()
features = glass_df.columns[1:-1]  # Exclude ID and target columns
glass_df[features] = scaler.fit_transform(glass_df[features])

# Split the data into training and testing sets
X = glass_df[features]
y = glass_df['Type of glass']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Logistic Regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
