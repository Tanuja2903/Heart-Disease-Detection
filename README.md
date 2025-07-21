# Heart Disease Detection using Machine Learning

## 🩺 Description

This project is a machine learning-based heart disease prediction system that analyzes patient data to determine the likelihood of heart disease. It leverages multiple classification algorithms including Logistic Regression, Support Vector Classifier (SVC), Random Forest, and K-Nearest Neighbors (KNN). The dataset is sourced from a standard heart disease dataset and contains various medical attributes such as age, cholesterol level, blood pressure, etc.

## 🛠️ Tech Stack

- Python
- Jupyter Notebook
- Pandas & NumPy (Data manipulation)
- Matplotlib & Seaborn (Visualization)
- Scikit-learn (Modeling and Evaluation)

## ✅ Features

- Data loading and preprocessing
- Exploratory Data Analysis (EDA) using visualization
- Feature scaling and train-test split
- Classification models: 
  - Logistic Regression
  - Support Vector Classifier (SVC)
  - Random Forest Classifier
  - K-Nearest Neighbors (KNN)
- Confusion matrix visualization for model evaluation
- Cross-validation for model accuracy

## 📸 Screenshots (Optional)

*You can add screenshots of your graphs, confusion matrix heatmap, or accuracy comparisons here.*

## 🚀 How to Run

1. **Clone the Repository** or open the `.ipynb` file in Jupyter Notebook.

2. **Install Dependencies** (if not already installed):
    ```bash
    pip install numpy pandas matplotlib seaborn scikit-learn
    ```

3. **Run the Notebook**:
    - Open `Heart Disease Detection Final.ipynb` in Jupyter.
    - Step through the notebook cells to view data processing, model training, and predictions.

4. **Test Prediction**:
    You can test the model using custom input data:
    ```python
    a = [[29,1,0,120,190,0,1,130,1,1.3,0,0,0]]
    kn.predict(a)  # Output: array([1])
    ```

    `1` indicates the presence of heart disease.

## 📂 Dataset Source

You need to have the dataset `heart.csv` located at:


Or modify the path according to your working environment.

---

*Author: Tanuja Shinde*  
*This project was built as a practical demonstration of machine learning applied to healthcare.*
