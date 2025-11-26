# train.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import skops.io as sio

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

# Ensure output folders exist
os.makedirs("Model", exist_ok=True)
os.makedirs("Results", exist_ok=True)

# 1. Load dataset
drug_df = pd.read_csv("Data/drug.csv")
drug_df = drug_df.sample(frac=1, random_state=1)  # shuffle for reproducibility

# 2. Split features/target
X = drug_df.drop("Drug", axis=1).values
y = drug_df["Drug"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=125
)

# 3. Construct preprocessing + model pipeline
cat_col = [1, 2, 3]  # columns indexes for categorical features from tutorial
num_col = [0, 4]    # numeric columns (age, Na_to_K) — adjust indices if your CSV differs

transform = ColumnTransformer(
    [
        ("encoder", OrdinalEncoder(), cat_col),
        ("num_imputer", SimpleImputer(strategy="median"), num_col),
        ("num_scaler", StandardScaler(), num_col),
    ]
)

pipe = Pipeline(
    steps=[
        ("preprocessing", transform),
        ("model", RandomForestClassifier(n_estimators=100, random_state=125)),
    ]
)

# 4. Train
pipe.fit(X_train, y_train)

# 5. Predict & evaluate
predictions = pipe.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions, average="macro")

print(f"Accuracy: {round(accuracy * 100, 2)}%  F1: {round(f1, 2)}")

# 6. Save metrics
with open("Results/metrics.txt", "w") as outfile:
    outfile.write(f"Accuracy = {round(accuracy, 4)}, F1 Score = {round(f1,4)}")

# 7. Confusion matrix
cm = confusion_matrix(y_test, predictions, labels=pipe.named_steps["model"].classes_ if hasattr(pipe.named_steps["model"], "classes_") else None)
if cm is None or cm.size == 0:
    # fallback: compute labels from y_test unique
    cm = confusion_matrix(y_test, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=getattr(pipe.named_steps['model'], 'classes_', None))
disp.plot()
plt.tight_layout()
plt.savefig("Results/model_results.png", dpi=120)
plt.close()

# 8. Save pipeline (skops)
sio.dump(pipe, "Model/drug_pipeline.skops")
print("Saved model to Model/drug_pipeline.skops")
