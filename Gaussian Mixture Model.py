import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.mixture import GaussianMixture

# Load dataset
data = pd.read_excel("Mess_Food_Quality_Dataset.xlsx")

# Encode categorical columns
le = LabelEncoder()
for col in ["Meal_Type", "Menu_Item", "Overall_Quality_Label"]:
    data[col] = le.fit_transform(data[col])

# Remove non-numeric column
X = data.drop(columns=["Date"])

# ✅ Standardize Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply Gaussian Mixture Model
gmm = GaussianMixture(n_components=3, random_state=42)
clusters = gmm.fit_predict(X_scaled)

# Add cluster labels to dataset
data["Cluster"] = clusters

# ✅ Print ALL rows
print("\nGMM Clustered Data (All Rows):\n")
print(data)

# Plot clusters (using first two standardized features)
plt.figure()
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters)
plt.title("Gaussian Mixture Model Clustering (Standardized Data)")
plt.xlabel("Standardized Feature 1")
plt.ylabel("Standardized Feature 2")
plt.show()