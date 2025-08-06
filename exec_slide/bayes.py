# Step 1: Import libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score

# Step 2: Training data
data = pd.DataFrame({
    'age': ['<=30', '<=30', '31...40', '>40', '>40', '>40', '31...40', '<=30', '<=30', '>40', '<=30', '31...40', '31...40', '>40'],
    'income': ['high', 'high', 'high', 'medium', 'low', 'low', 'low', 'medium', 'low', 'medium', 'medium', 'medium', 'high', 'medium'],
    'student': ['no', 'no', 'no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no'],
    'credit_rating': ['fair', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'excellent'],
    'buys_computer': ['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'no']
})

# Step 3: Encode categorical variables
encoders = {}
for col in ['age', 'income', 'student', 'credit_rating', 'buys_computer']:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    encoders[col] = le

# Step 4: Split features and target
X = data[['age', 'income', 'student', 'credit_rating']]
y = data['buys_computer']

# Step 5: Train Naive Bayes model
model = CategoricalNB()
model.fit(X, y)

# Step 6: Predict on training data
predictions = model.predict(X)
decoded_preds = encoders['buys_computer'].inverse_transform(predictions)

# Step 7: Display predictions
print("🔍 Predictions on training data:")
for i, pred in enumerate(decoded_preds, start=1):
    print(f"RID {i}: {pred}")

# Step 8: Accuracy
accuracy = accuracy_score(y, predictions)
print(f"\n✅ Accuracy on training data: {accuracy:.2f}")

# Step 9: Predict new sample
new_sample = pd.DataFrame({
    'age': ['<=30'],
    'income': ['medium'],
    'student': ['yes'],
    'credit_rating': ['fair']
})

# Encode new sample
for col in new_sample.columns:
    new_sample[col] = encoders[col].transform(new_sample[col])

# Predict
new_pred = model.predict(new_sample)
decoded_pred = encoders['buys_computer'].inverse_transform(new_pred)

# Show result
print(f"\n🧠 Prediction for new sample: {decoded_pred[0]}")
