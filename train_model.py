import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATA_PATH = "simulated_irrigation_data.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
features = ["temperature_c", "humidity_pct", "light_lux", "soil_moisture_pct"]
X = df[features]
y = df["irrigation_required"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=300, max_depth=12, min_samples_leaf=3,
    class_weight="balanced", random_state=42, n_jobs=-1
)
model.fit(X_train, y_train)

pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, pred):.3f}")
print("Confusion matrix:\\n", confusion_matrix(y_test, pred))
print(classification_report(y_test, pred, target_names=["No irrigation", "Irrigation"]))

joblib.dump({"model": model, "features": features}, f"{MODEL_DIR}/irrigation_model.joblib")
print("Saved model to models/irrigation_model.joblib")
