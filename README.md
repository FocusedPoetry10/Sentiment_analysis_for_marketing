# Sentiment Analysis for Marketing
*October 30, 2023*

## Overview

### 1. Project Overview
This project aims to develop a sentiment analysis model for marketing purposes. The model will be trained on a dataset of tweets about US airlines and will be used to predict the sentiment of new tweets. This information can then be used by businesses to improve their marketing campaigns and customer service.

### 2. Dataset
The dataset used in this project is the Twitter Airline US Sentiment dataset from Kaggle. This dataset contains over 14,000 tweets about US airlines, each labeled with its sentiment (positive, negative, or neutral).

### 3. Data Preprocessing
The following data preprocessing steps were performed on the dataset:
- Removed punctuation and stop words.
- Lemmatized the text.
- Converted the text to lowercase.

### 4. Feature Extraction
The following features were extracted from the preprocessed text:
- Bag of Words (BOW) features
- TF-IDF features

### 5. Machine Learning Algorithm
Two machine learning algorithms were used to train the sentiment analysis model:
- Random Forest Classifier
- Support Vector Machine (SVM)

### 6. Model Training
The following steps were involved in training the machine learning models:
- Split the dataset into training and testing sets.
- Train the models on the training set.
- Evaluate the models on the testing set.

### 7. Evaluation Metrics
The following evaluation metrics were used to assess the performance of the machine learning models:
- Accuracy
- Precision
- Recall
- F1 Score

### 8. Innovative Techniques
The following innovative techniques were used during the development of the sentiment analysis model:
- A hybrid approach was used to extract features from the text. This approach combined BOW features with TF-IDF features, resulting in better performance than using either type of feature alone.
- A hyperparameter tuning algorithm was used to optimize the parameters of the machine learning models, resulting in significant improvements in accuracy.

### 9. Dataset Source and Description
The Twitter Airline US Sentiment dataset is available on Kaggle at the following link: [Twitter Airline US Sentiment Dataset](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment).

This dataset contains over 14,000 tweets about US airlines, each labeled with its sentiment (positive, negative, or neutral). The tweets were collected in 2015 and represent a diverse range of opinions about US airlines.

### 10. How to Run the Code
The following steps can be used to run the code for this project:
1. Clone the repository.
2. Install the required dependencies.
3. Run the data preprocessing script.
4. Run the feature extraction script.
5. Train the machine learning models.
6. Evaluate the machine learning models.

### 11. Dependencies
The following dependencies are required to run the code for this project:
- Python 3.6+
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

### 12. Conclusion
This project developed a sentiment analysis model for marketing purposes. The model is trained on a dataset of tweets about US airlines and can be used to predict the sentiment of new tweets. This information can then be used by businesses to improve their marketing campaigns and customer service.
