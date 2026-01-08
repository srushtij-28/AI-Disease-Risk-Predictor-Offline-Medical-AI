import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Sample patient dataset
data = {
    "age": [25, 45, 35, 60, 50, 30],
    "bp": [120, 140, 130, 150, 145, 125],
    "cholesterol": [180, 220, 200, 240, 230, 190],
    "risk": [0, 1, 0, 1, 1, 0]   # 0 = Low risk, 1 = High risk
}

df = pd.DataFrame(data)

X = df[["age", "bp", "cholesterol"]]
y = df["risk"]

model = DecisionTreeClassifier()
model.fit(X, y)

print("🩺 Disease Risk Predictor \n")

age = int(input("Enter age: "))
bp = int(input("Enter blood pressure: "))
chol = int(input("Enter cholesterol level: "))

pred = model.predict([[age, bp, chol]])[0]

if pred == 1:
    print("\n⚠️ High Risk — Consult a doctor")
else:
    print("\n✅ Low Risk — Stay healthy")
