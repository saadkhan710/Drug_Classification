# Drug Classification Using Machine Learning

## Project Overview

This project applies supervised machine learning techniques to classify drug types based on patient medical data. The goal is to build an intelligent system that can assist in preliminary drug recommendation by learning from key patient features.

---

## Dataset

- **Source:** [Kaggle - Drug Classification Dataset](https://www.kaggle.com/datasets/ibrahimbahbah/drug200/data)
- **Features Include:**
  - Age
  - Gender
  - Blood Pressure (BP)
  - Cholesterol
  - Sodium-to-Potassium Ratio (Na to K)
  - Drug Type (Target Variable with 5 categories)

---

## Objectives

- Predict drug type based on patient’s medical attributes.
- Address class imbalance through appropriate evaluation metrics.
- Compare performance across multiple models.
- Optimize and interpret the final model.
- Highlight limitations and suggest future improvements.

---

## Models Explored

- Logistic Regression  
- Decision Tree  
- Random Forest  
- K-Nearest Neighbors  
- Support Vector Machine (SVM)

---

## Final Model

- **Selected Model:** Support Vector Machine (SVM)
- **Hyperparameter Tuning:** GridSearchCV
- **Best Parameters:** `C=10`, `gamma='scale'`, `class_weight='balanced'`
- **F1 Score on Test Set:** 0.93

---

## Key Highlights

- **Class Imbalance:** Emphasized metrics like Precision, Recall, and F1 Score over simple Accuracy.
- **Model Evaluation:** Used Train/Validation/Test split to avoid overfitting.
- **Error Analysis:** Confusion matrix used to understand misclassifications (notably between DrugX and DrugY).
- **Interpretability:** Discussed the need for explainable AI, especially in healthcare applications.
- **Ethical Considerations:** Reflected on fairness, data limitations, and potential biases in the model.

---

## Future Work

- Incorporate more medical features like weight, diabetes history, and prior conditions.
- Use explainability tools like SHAP or LIME to better understand model decisions.
- Collaborate with healthcare professionals to validate predictions and interpret results.
- Scale up with real-world datasets for broader generalization.

---

## 🛠How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/saadkhan710/Drug_Classification.git
   cd Drug_Classification

   ```
 2. Install dependencies:
  ``` bash
   pip install -r requirements.txt

  ````
 3. Run the notebook:
    
    Open Drug_ML.ipynb in Jupyter Notebook or VS Code.

 4. Load the final model (optional):

  ``` bash
import pickle
with open("best_svm_model.pkl", "rb") as f:
    model = pickle.load(f)
predictions = model.predict(X_test_prepared)
 ```

## Acknowledgements

- UCD MSc Information Systems Faculty
- ADAPT Centre (Inspiration from PRECISE4Q Stroke AI initiative)
- Kaggle Dataset Providers

  


   
