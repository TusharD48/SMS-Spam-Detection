# SMS-Spam-Detection
[![MasterHead](https://editor.analyticsvidhya.com/uploads/32086heading.jpeg)](https://ww38.rishavchanda.io/)

# Table of Content
-  Introduction
- Project Structure
- Installation
- Usage
- Dataset
- Model
- Results
- Contributing
- License
- Contact

# Introduction
This project aims to build a machine learning model to detect spam messages in SMS. Spam detection is crucial for filtering out unwanted and potentially harmful messages, enhancing user experience, and improving communication security.

# Project Structure
The project directory is structured as follows:
```
sms-spam-detection/
├── data/
│   ├── raw/
│   ├── processed/
├── notebooks/
│   ├── sms-spam-detection.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── spam_detection.py
├── models/
├── results/
├── README.md

```

## Installation

```
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
pip install pandas numpy scikit-learn nltk xgboost
```

# Model and Technique
This project utilizes several machine learning techniques and models to perform spam detection:

- Logistic Regression
- Support Vector Classifier (SVC)
- Multinomial Naive Bayes (MultinomialNB)
- Decision Tree Classifier
- K-Neighbors Classifier (KNeighborsClassifier)
- Random Forest Classifier
- AdaBoost Classifier
- Bagging Classifier
- Extra Trees Classifier
- Gradient Boosting Classifier
- XGBoost Classifier

Results
The performance of the spam detection model is evaluated using metrics such as accuracy, precision, recall, and F1-score. The results are stored in the results/ directory. Here is a summary of the model's performance:

| Algorithm | Accuracy | Precision |
|-----------|----------|-----------|
| KN        | 0.904255 | 1.000000  |
| NB        | 0.972921 | 0.991597  |
| ETC       | 0.977756 | 0.984127  |
| RF        | 0.971954 | 0.975410  |
| GBDT      | 0.946809 | 0.968750  |
| SVC       | 0.976789 | 0.954887  |
| xgb       | 0.968085 | 0.937500  |
| BgC       | 0.962282 | 0.884058  |
| LR        | 0.946809 | 0.868852  |
| AdaBoost  | 0.950677 | 0.867188  |
| DT        | 0.933269 | 0.827586  |
