import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv('heartanalysis.csv')

# Split data
X = data.drop('target', axis=1)
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define models
rf = RandomForestClassifier()
gb = GradientBoostingClassifier()
model = VotingClassifier(estimators=[('rf', rf), ('gb', gb)], voting='soft')

# Train and test
model.fit(X_train, y_train)
pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test,pred))