import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans

# Load dataset
data = pd.read_excel("Mess_Food_Quality_Dataset.xlsx")

# Convert categorical columns
le = LabelEncoder()
for col in ["Meal_Type", "Menu_Item", "Overall_Quality_Label"]:
    data[col] = le.fit_transform(data[col])

# Remove non-numeric column
X = data.drop(columns=["Date"])

# ✅ Standardize Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Add cluster labels to dataset
data["Cluster"] = clusters

# ✅ Print ALL rows
print("\nClustered Data (All Rows):\n")
print(data)

# Plot clusters (using first two standardized features)
plt.figure()
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters)
plt.title("K-Means Clustering of Mess Food Data (Standardized)")
plt.xlabel("Standardized Feature 1")
plt.ylabel("Standardized Feature 2")
plt.show()