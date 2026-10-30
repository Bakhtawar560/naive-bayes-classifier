import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import (accuracy_score, classification_report,
                             ConfusionMatrixDisplay)

# ============================================================
# PART 1: Gaussian Naive Bayes on the Iris dataset (numeric data)
# ============================================================
print("=" * 55)
print("PART 1: GaussianNB on Iris")
print("=" * 55)

iris = load_iris(as_frame=True)
X, y = iris.data, iris.target

# Split: 70% to learn from, 30% to test on (data the model has never seen)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print("Training rows:", len(X_train), "| Testing rows:", len(X_test))

model = GaussianNB()
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("Accuracy on unseen test data:", round(accuracy_score(y_test, pred) * 100, 2), "%")
print("\nClassification report:")
print(classification_report(y_test, pred, target_names=iris.target_names))

# Save the confusion matrix as a picture
ConfusionMatrixDisplay.from_predictions(
    y_test, pred, display_labels=iris.target_names, cmap="Blues"
)
plt.title("Gaussian Naive Bayes - Iris")
plt.savefig("iris_confusion_matrix.png", dpi=150, bbox_inches="tight")
print("Saved iris_confusion_matrix.png")

# ============================================================
# PART 2: Categorical Naive Bayes on Play Tennis (text categories)
# ============================================================
print("\n" + "=" * 55)
print("PART 2: CategoricalNB on Play Tennis")
print("=" * 55)

df = pd.read_csv("data/play_tennis.csv")
features = df.drop("Decision", axis=1)
target = df["Decision"]

# Naive Bayes needs numbers, so turn words (Sunny, Rain...) into codes
encoder = OrdinalEncoder()
features_encoded = encoder.fit_transform(features)

cnb = CategoricalNB(alpha=1.0)   # alpha=1 is Laplace smoothing
cnb.fit(features_encoded, target)

train_acc = accuracy_score(target, cnb.predict(features_encoded))
print("Accuracy on the 14 training rows:", round(train_acc * 100, 2), "%")

# Predict a new day
new_day = pd.DataFrame([["Sunny", "Cool", "Normal", "Weak"]], columns=features.columns)
new_encoded = encoder.transform(new_day)
print("\nNew day:", list(new_day.iloc[0]))
print("Prediction:", cnb.predict(new_encoded)[0])
for label, p in zip(cnb.classes_, cnb.predict_proba(new_encoded)[0]):
    print("  Probability of", label, "=", round(p * 100, 2), "%")