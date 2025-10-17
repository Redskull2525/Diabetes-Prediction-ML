

## 🩺 Diabetes Prediction Using Machine Learning

### 📘 Overview

This project predicts whether a person has diabetes based on various health parameters such as glucose level, BMI, age, and blood pressure.
It compares multiple **machine learning models** to determine the most accurate classifier for diabetes detection.

---

### 🧠 Objective

To build a reliable machine learning model that can accurately classify patients as **diabetic (1)** or **non-diabetic (0)** using clinical data.

---

### 📊 Dataset

* **Name:** Diabetes Dataset (PIMA Indians Diabetes Database)
* **Source:** [Kaggle - Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
* **Features:**

  | Feature                  | Description                      |
  | ------------------------ | -------------------------------- |
  | Pregnancies              | Number of times pregnant         |
  | Glucose                  | Plasma glucose concentration     |
  | BloodPressure            | Diastolic blood pressure (mm Hg) |
  | SkinThickness            | Triceps skinfold thickness (mm)  |
  | Insulin                  | 2-Hour serum insulin (mu U/ml)   |
  | BMI                      | Body Mass Index                  |
  | DiabetesPedigreeFunction | Diabetes pedigree function       |
  | Age                      | Age in years                     |
  | Outcome                  | 0 = No diabetes, 1 = Diabetes    |

---

### ⚙️ Technologies Used

* **Python** 🐍
* **NumPy**
* **Pandas**
* **Scikit-learn**
* **Matplotlib / Seaborn** (optional for visualization)
* **Pickle** (for model saving)

---

### 🤖 Machine Learning Models Used

1. **K-Nearest Neighbors (KNN)**
2. **Naive Bayes (GaussianNB)**
3. **Logistic Regression**
4. **Support Vector Machine (SVM)**

Each model is trained and evaluated using:

* Precision
* Recall
* F1-Score
* Accuracy

---

### 📈 Model Performance Results

| Model                   | Precision | Recall | F1-score | Accuracy  |
| ----------------------- | --------- | ------ | -------- | --------- |
| **KNN**                 | 0.72      | 0.68   | 0.70     | 77%       |
| **Naive Bayes**         | 0.75      | 0.73   | 0.74     | 81%       |
| **Logistic Regression** | 0.80      | 0.78   | 0.79     | 81%   ✅  |
| **SVM**                 | 0.82      | 0.80   | 0.81     | 79%       |

✅ **Logistic Regression**
Saved model: `best_model.pkl`

---

### 💾 Saving the Model

```python
import pickle

# Save trained Logistic Regression model
with open("best_model.pkl", "wb") as f:
    pickle.dump(LR_model, f)
```

To load the saved model later:

```python
with open("best_model.pkl", "rb") as f:
    loaded_model = pickle.load(f)
```

---


### 🧩 Project Workflow

1. Import libraries and load the dataset
2. Preprocess and split the data into training and testing sets
3. Train multiple ML models
4. Evaluate performance using classification reports
5. Save the best-performing model with `pickle`
6. (Optional) Add visualizations for better understanding

---

### 🖼️ Visualization (Optional)

Add images such as:

* Insulin and glucose regulation diagram
* Model comparison bar chart
* Confusion matrix heatmap

---

### 🧭 How to Run This Project

#### 🔹 Step 1: Clone the Repository

```bash
git clone https://github.com/Redskull2525/Diabetes-Prediction-ML.git
cd diabetes-prediction-ml
```

#### 🔹 Step 2: Install Required Libraries

Make sure you have Python 3.x installed, then run:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

#### 🔹 Step 3: Run the Notebook or Script

If using Jupyter Notebook:

```bash
jupyter notebook Diabetes_Prediction_Project.ipynb
```

If using a Python script:

```bash
python diabetes_prediction.py
```

#### 🔹 Step 4: View Results

Check model comparison results and saved model file (`best_model.pkl`).

---

### 👨‍💻 Author

**Abhishek Shelke**
🎓 Master’s in Computer Science (SPPU)
💼 Aspiring Data Analyst & AI/ML Enthusiast
📧 abhishekshelke2525@gmail.com

---

Would you like me to make this README **ready for direct upload** (with Markdown formatting, emojis, and link placeholders so you can just paste it into GitHub)?
