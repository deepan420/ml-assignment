import os
print("Files in current folder:", os.listdir())

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import plot_tree

# Step 1: Load Dataset
data = pd.read_excel("Mess_Food_Quality_Dataset.xlsx")

# Step 2: Convert Categorical Data to Numbers
le = LabelEncoder()
for col in ["Meal_Type", "Menu_Item", "Overall_Quality_Label"]:
    data[col] = le.fit_transform(data[col])

# Step 3: Split Features and Target
X = data.drop(columns=["Date", "Overall_Quality_Label"])
y = data["Overall_Quality_Label"]

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 5: Train Random Forest Model
rf_model = RandomForestClassifier(
    n_estimators=50,
    max_depth=5,
    random_state=42
)
rf_model.fit(X_train, y_train)

# Step 6: Make Predictions
y_pred = rf_model.predict(X_test)

# Step 7: Create Prediction Table
results = pd.DataFrame({
    "Actual Quality": y_test,
    "Predicted Quality": y_pred
})

# Convert numeric labels back to text
label_map = {0: "Poor", 1: "Average", 2: "Good"}
results["Actual Quality"] = results["Actual Quality"].map(label_map)
results["Predicted Quality"] = results["Predicted Quality"].map(label_map)

print("\nPrediction Table:\n")
print(results)

# Step 8: Accuracy
print("\nAccuracy:", accuracy_score(y_test, y_pred))

# Step 9: Confusion Matrix
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Step 10: Prediction Table as Figure
fig, ax = plt.subplots(figsize=(8, 6))
ax.axis('off')
table = ax.table(
    cellText=results.values,
    colLabels=results.columns,
    loc='center'
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.auto_set_column_width(col=list(range(len(results.columns))))
plt.title("Random Forest Prediction Results")
plt.show()

# Step 11: 🌲 Display Trees One-by-One
num_trees_to_plot = 5   # change this number if needed

for i in range(num_trees_to_plot):
    plt.figure(figsize=(18,10))
    plot_tree(
        rf_model.estimators_[i],
        feature_names=X.columns,
        class_names=["Poor", "Average", "Good"],
        filled=True
    )
    plt.title(f"Random Forest - Tree {i+1}")
    plt.show()