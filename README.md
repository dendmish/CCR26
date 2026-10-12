# Customer Churn Prediction

Machine Learning project for predicting customer churn based on customer behavior and service usage patterns.
https://dendmish.pythonanywhere.com/
## Project Objective

The goal of this project is to predict whether a customer is likely to churn using machine learning methods and deploy the final model as a FastAPI microservice.

---

## Dataset

Source: Kaggle Customer Churn Dataset

The dataset contains information about customer activity, support interactions, payment behavior, and account history.

### Features Used

- Tenure
- Support Calls
- Payment Delay

### Target Variable

- Churn
    - 0 - Customer stays
    - 1 - Customer leaves

---

## Exploratory Data Analysis

The following analyses were performed:

- Feature distribution analysis
- Histograms and boxplots
- Churn class analysis
- Comparison of customer groups
- Investigation of model prediction errors

---

## Models Evaluated

The following models were trained and compared:

- Logistic Regression
- Polynomial Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Random Forest
- LightGBM

### Best Model

LightGBM demonstrated the best predictive performance and was selected for deployment.

---
## Final Performance

| Metric | Value |
|----------|----------|
| ROC-AUC | 0.9313 |


## Model Persistence

The final trained model was saved as a serialized Scikit-Learn Pipeline:

```python
joblib.dump(final_pipeline, "churn_pipeline.pkl")
```

Saved artifact:

```text
churn_pipeline.pkl
```

---

## FastAPI Deployment

A FastAPI microservice was implemented for model inference.

### Run API

```bash
py -m uvicorn main:app --reload
```

### Swagger Documentation

```text
http://127.0.0.1:8000/docs
```

---

## Example Request

```json
{
  "Tenure": 20,
  "Support_Calls": 5,
  "Payment_Delay": 10
}
```

### Example Response

```json
{
  "prediction": 0,
  "churn_probability": 0.12691610104552536
}
```

---

## Project Structure

```text
CCR26/
│
├── CCR_2026.ipynb
├── main.py
├── churn_pipeline.pkl
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Dependencies

- Python
- Pandas
- NumPy
- Scikit-Learn
- LightGBM
- FastAPI
- Joblib
- Matplotlib
- Seaborn

---

## Author

@dendmish (Aleksandr Denisov)
