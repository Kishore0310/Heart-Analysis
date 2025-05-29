# ❤️ Heart Disease Prediction

This project uses machine learning to predict the likelihood of heart disease in patients based on medical attributes. It leverages ensemble methods — combining **Random Forest** and **Gradient Boosting** classifiers using a **VotingClassifier** with soft voting.

## 📂 Dataset

The dataset used is `heartanalysis.csv`. It should contain the following columns (example features):

- `age`
- `sex`
- `cp` (chest pain type)
- `trestbps` (resting blood pressure)
- `chol` (serum cholesterol)
- `fbs` (fasting blood sugar)
- `restecg` (resting electrocardiographic results)
- `thalach` (maximum heart rate achieved)
- `exang` (exercise-induced angina)
- `oldpeak` (ST depression induced by exercise)
- `slope`, `ca`, `thal`
- `target` (0 = no disease, 1 = disease)

## 🛠️ Installation

Install the required packages using pip:

```bash
pip install pandas scikit-learn
