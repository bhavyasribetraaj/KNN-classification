# K-Nearest Neighbors (KNN) - Iris Flower Classification

## 📌 Task Overview
 
The objective is to **implement the K-Nearest Neighbors algorithm** for classification, experiment with different values of K, normalize features, and visualize decision boundaries.

The implementation is done as an **interactive Streamlit dashboard** for better visualization and experimentation.

---

## 📂 Dataset
I used the **Iris Dataset** from UCI/Kaggle.  

- **Target variable:** `Species` (Iris-setosa, Iris-versicolor, Iris-virginica)  
- **Features:** 4 numerical measurements of iris flowers  
- **Shape:** 150 rows × 5 columns  

**Dataset Link:**  
[Iris Dataset - Kaggle](https://www.kaggle.com/datasets/uciml/iris)  

---

## ⚙️ Steps Performed
1. **Data Loading**
   - Load `Iris.csv` from the given path  
   - Drop `Id` column if present

2. **Feature Normalization**
   - Standardized features using `StandardScaler` for better distance computation

3. **Label Encoding**
   - Converted string labels into numeric values for KNN & decision boundary plotting

4. **Train/Test Split**
   - User-controlled split via slider (default 80% training, 20% testing)

5. **Model Building**
   - **KNeighborsClassifier**
     - Adjustable `n_neighbors` (K), distance metric, and weighting scheme via sidebar controls

6. **Model Evaluation**
   - Accuracy score
   - Confusion matrix (heatmap)
   - Classification report

7. **Parameter Analysis**
   - Accuracy vs K plot for optimal K selection

8. **Decision Boundaries**
   - PCA projection to 2D for visualization  
   - Colored background shows model-predicted regions for each class  
   - Scatter points show actual samples

---

## 📊 Example Output

**=== Model Accuracy ===**  
KNN Accuracy: 96.67%  

**Confusion Matrix**  
```
[[10  0  0]
 [ 0  9  1]
 [ 0  1  9]]
```

**Classification Report**  
```
                 precision    recall  f1-score   support
Iris-setosa       1.00       1.00      1.00        10
Iris-versicolor   0.90       0.90      0.90        10
Iris-virginica    0.90       0.90      0.90        10
```

---

## 📈 Visualizations
- Confusion matrix (heatmap)  
- Accuracy vs K plot  
- Decision boundaries with PCA projection  

---

## 📦 Requirements
```bash
pip install streamlit pandas scikit-learn seaborn matplotlib numpy
```

---

## ▶ How to Run
```bash
python -m streamlit run KNN_classification.py
```
Then open the local URL provided in the terminal.
