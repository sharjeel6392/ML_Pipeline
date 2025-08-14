# End-to-End MLOps Pipeline for Spam Detection
This project demonstrates the creation of a complete and reproducible MLOps pipeline for spam detection. The primary objective was to build a system that can accurately classify text messages as "spam" or "ham" and manage the entire machine learning workflow using MLOps best practices.

## Problem Statement

The core challenge was to effectively filter unsolicited and malicious messages from legitimate ones. This required developing a high-performance classification model and automating the process of data versioning, model training, and deployment to ensure a robust and scalable solution.

## Data

The project utilized a labeled dataset of English text messages:

Training Data: 4704 entries, each consisting of a text message and a corresponding "spam" or "ham" target label.

Testing Data: 870 entries for independent model evaluation.

## Methodology

### ML Model Development:

### Text Preprocessing: 
Used NLTK to import stopwords and the punkt tokenizer. The text was cleaned, stemmed using PorterStemmer, and the target labels were encoded using LabelEncoder.

### Feature Engineering: 
The preprocessed text was transformed into numerical features using TF-IDF (Term Frequency-Inverse Document Frequency) vectorization.

### Model Selection: 
A wide range of machine learning classifiers was trained and evaluated, including Logistic Regression, SVM, Multinomial Naive Bayes, Decision Tree, KNN, and various ensemble methods. Based on a comparison of accuracy and precision, the Random Forest Classifier was selected as the final model due to its superior performance.

## MLOps Pipeline Implementation:

### Data and Model Versioning: 
The project leveraged DVC (Data Version Control) to manage and track different versions of the dataset and the trained models. This ensures that the entire pipeline is reproducible, and any changes to the data or code are easily traceable.

### Cloud Storage: 
AWS S3 was used as the remote storage backend for DVC, securely hosting all data, models, and artifacts. This provides a centralized and scalable repository for the project's assets.

### Pipeline Automation: 
The entire workflow—from data preprocessing and feature engineering to model training and evaluation—was orchestrated using DVC pipelines, ensuring that the process can be easily re-run and automated.

# Tools & Technologies:

## ML & Data Science: 
NLTK, Scikit-learn, Pandas, NumPy

## MLOps & DevOps: 
DVC (Data Version Control), AWS S3

## Models: 
Random Forest Classifier (final model), SVM, Naive Bayes, etc.