# Importing Libraries
import pandas as pd
import re
import nltk
import logging
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder

# Setting up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("Starting the NLP Model for Text Classification")

# Load the Datasets
logging.info("Loading datasets...")
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
logging.info(f"Train Set shape: {train_df.shape}")
logging.info(f"Test Set shape: {test_df.shape}")

# Data Preprocessing
logging.info("Starting data preprocessing...")
def preprocess_text(text):
    text = re.sub(r'\W', ' ', str(text))  # Remove special characters
    text = text.lower()  # Lowercase
    text = text.split()
    text = [word for word in text if word not in stopwords.words('english')]  # Remove stopwords
    text = ' '.join(text)
    return text

# Applying Preprocessing
logging.info("Applying text preprocessing to training and test sets...")
train_df['clean_text'] = train_df['crimeaditionalinfo'].apply(preprocess_text)
test_df['clean_text'] = test_df['crimeaditionalinfo'].apply(preprocess_text)
logging.info("Data preprocessing completed.")

# Label Encoding for Categories and Subcategories
logging.info("Encoding labels for categories and subcategories...")
label_encoder = LabelEncoder()
train_df['category_encoded'] = label_encoder.fit_transform(train_df['category'])
train_df['subcategory_encoded'] = label_encoder.fit_transform(train_df['sub_category'])
logging.info("Label encoding completed.")

# Feature Extraction using TF-IDF
logging.info("Extracting features using TF-IDF...")
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(train_df['clean_text'])
X_test_tfidf = tfidf.transform(test_df['clean_text'])
logging.info("TF-IDF feature extraction completed.")

# Model Training
logging.info("Training the Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_tfidf, train_df['category_encoded'])
logging.info("Model training completed.")

# Prediction on Test Set
logging.info("Generating predictions on the test set...")
y_pred = model.predict(X_test_tfidf)

# Converting Predicted Labels back to Original Category Names
y_pred_labels = label_encoder.inverse_transform(y_pred)
logging.info("Predictions completed.")

all_categories = pd.concat([train_df['category'], test_df['category']]).unique()

# Fit the LabelEncoder on all unique categories
label_encoder = LabelEncoder()
label_encoder.fit(all_categories)

# Encode training and test categories
train_df['category_encoded'] = label_encoder.transform(train_df['category'])

# For test set: Check if category column exists, then transform
if 'category' in test_df.columns:
    try:
        test_df['category_encoded'] = label_encoder.transform(test_df['category'])
    except ValueError as e:
        print(f"Error in encoding test categories: {e}")
else:
    print("No 'category' column in test set; skipping encoding for test categories.")

# Export Predictions to CSV if needed
logging.info("Saving predictions to CSV...")
test_df['predicted_category'] = y_pred_labels
test_df.to_csv("test_predictions.csv", index=False)
logging.info("Predictions saved to 'test_predictions.csv'.")

logging.info("Process completed.")
