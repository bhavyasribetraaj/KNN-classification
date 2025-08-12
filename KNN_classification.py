import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA

st.set_page_config(page_title="KNN Classifier - Task 6", layout="wide")
st.title("🌸 K-Nearest Neighbors (KNN) Classification - Iris Dataset")

df = pd.read_csv(r"C:\Users\Betraaj\Downloads\Desktop\elevate labs\TASK 6\Iris.csv")

st.subheader("Dataset Preview")
st.dataframe(df.head())

if "Id" in df.columns:
    df.drop("Id", axis=1, inplace=True)

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

test_size = st.sidebar.slider("Test Size (%)", 10, 50, 20, 5) / 100
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=test_size, random_state=42)

k_value = st.sidebar.slider("Number of Neighbors (K)", 1, 20, 5)
metric_choice = st.sidebar.selectbox("Distance Metric", ["minkowski", "euclidean", "manhattan"])
weights_choice = st.sidebar.selectbox("Weight Function", ["uniform", "distance"])

knn = KNeighborsClassifier(n_neighbors=k_value, metric=metric_choice, weights=weights_choice)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

acc = accuracy_score(y_test, y_pred)

st.subheader("📊 Model Accuracy")
st.metric("KNN Accuracy", f"{acc:.2%}")

st.subheader("Confusion Matrix")
fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
st.pyplot(fig)

st.subheader("Classification Report")
st.text(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

st.subheader("K Value vs Accuracy")
k_range = range(1, 21)
scores = []
for k in k_range:
    model = KNeighborsClassifier(n_neighbors=k, metric=metric_choice, weights=weights_choice)
    model.fit(X_train, y_train)
    scores.append(accuracy_score(y_test, model.predict(X_test)))
fig, ax = plt.subplots()
ax.plot(k_range, scores, marker="o")
ax.set_xlabel("Number of Neighbors (K)")
ax.set_ylabel("Accuracy")
st.pyplot(fig)

st.subheader("Decision Boundaries (2D Projection)")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y_encoded, test_size=test_size, random_state=42)
knn_pca = KNeighborsClassifier(n_neighbors=k_value, metric=metric_choice, weights=weights_choice)
knn_pca.fit(X_train_pca, y_train_pca)

x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

Z = knn_pca.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

fig, ax = plt.subplots()
ax.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=label_encoder.inverse_transform(y_encoded),
                palette="deep", ax=ax, edgecolor="k")
plt.title("Decision Boundaries (PCA Projection)")
st.pyplot(fig)
