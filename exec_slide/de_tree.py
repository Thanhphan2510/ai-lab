# Step 1: Import necessary libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# Step 2: Training data
train_data = pd.DataFrame({
    'time': ['1-2', '2-7', '>7', '1-2', '>7', '1-2', '2-7', '2-7'],
    'gender': ['m', 'm', 'f', 'f', 'm', 'm', 'f', 'm'],
    'area': ['urban', 'rural', 'rural', 'rural', 'rural', 'rural', 'urban', 'urban'],
    'risk': ['low', 'high', 'low', 'high', 'high', 'high', 'low', 'low']
})

# Step 3: Encode categorical features
encoders = {}
for col in ['time', 'gender', 'area']:
    le = LabelEncoder()
    train_data[col] = le.fit_transform(train_data[col])
    encoders[col] = le

# Encode target variable
le_risk = LabelEncoder()
train_data['risk'] = le_risk.fit_transform(train_data['risk'])

# Step 4: Train Decision Tree model
X_train = train_data[['time', 'gender', 'area']]
y_train = train_data['risk']
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Step 5: New samples to classify
new_samples = pd.DataFrame({
    'time': ['1-2', '2-7', '1-2'],
    'gender': ['f', 'm', 'f'],
    'area': ['rural', 'urban', 'urban']
}, index=['A', 'B', 'C'])

# Step 6: Encode new samples using same encoders
for col in ['time', 'gender', 'area']:
    new_samples[col] = encoders[col].transform(new_samples[col])

# Step 7: Predict risk levels
predictions = model.predict(new_samples)
decoded_predictions = le_risk.inverse_transform(predictions)

# Step 8: Display results
print("🔍 Predicted risk levels for new samples:")
for idx, pred in zip(new_samples.index, decoded_predictions):
    print(f"ID {idx}: {pred}")
