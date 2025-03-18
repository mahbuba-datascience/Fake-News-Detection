Fake News Detection Project
Project Overview
This project aims to detect fake news articles using machine learning techniques. It involves data preprocessing, vectorization, feature extraction, and classification using multiple machine learning models to determine the authenticity of news articles.

Dataset
The dataset contains news articles labeled as Fake or True. It includes:

title: The headline of the news article
text: The main content of the article
subject: The category of the news
date: The publication date
type: Label indicating whether the news is Fake or True
Key Features
Text Preprocessing: Tokenization, stopword removal, lemmatization, and vectorization using spaCy and NLTK.
Exploratory Data Analysis: Word cloud visualizations, sentiment analysis, and topic modeling.
Feature Engineering: Word embeddings using Word2Vec, topic modeling with LDA.
Machine Learning Models: Logistic Regression, Random Forest, Decision Tree, Support Vector Machine (SVM), and Gradient Boosting Classifier for classification.
Evaluation Metrics: Accuracy, precision, recall, F1-score, and confusion matrix visualization.
Installation
Ensure you have the required dependencies installed:

bash
Copy
Edit
pip install numpy pandas matplotlib seaborn nltk spacy gensim scikit-learn wordcloud pyLDAvis
python -m spacy download en_core_web_md
Usage
Run the Jupyter Notebook (Final_Fake_News_Detection_Project.ipynb) to execute the complete pipeline:

bash
Copy
Edit
jupyter notebook Final_Fake_News_Detection_Project.ipynb
Results
Best Model Performance:
Random Forest achieved the highest accuracy (~96%) in distinguishing fake news from real news.
SVM also performed well with a similar accuracy.
Future Improvements
Implement deep learning models (LSTMs, Transformers).
Explore additional NLP techniques for feature extraction.
Incorporate real-time fake news detection using web scraping.
