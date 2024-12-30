import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import LatentDirichletAllocation
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import warnings

warnings.filterwarnings('ignore')  # Suppress warnings

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Load datasets
train_data = pd.read_csv('E:\\sentiment140\\trainingdata.csv', encoding='ISO-8859-1', header=None)
train_data.columns = ['sentiment', 'id', 'date', 'query', 'user', 'text']

test_data = pd.read_csv('E:\\sentiment140\\testdata.csv', encoding='ISO-8859-1', header=None)
test_data.columns = ['sentiment', 'id', 'date', 'query', 'user', 'text']

# Drop unnecessary columns
train_data = train_data[['sentiment', 'text']]
test_data = test_data[['sentiment', 'text']]

# Map sentiments to three classes (0=Negative, 2=Neutral, 4=Positive)
train_data['sentiment'] = train_data['sentiment'].map({0: 0, 2: 1, 4: 2})
test_data['sentiment'] = test_data['sentiment'].map({0: 0, 2: 1, 4: 2})

# Remove missing values
train_data = train_data.dropna()
test_data = test_data.dropna()

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Preprocessing function
def preprocess(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)    # Remove mentions
    text = re.sub(r'#\w+', '', text)    # Remove hashtags
    text = re.sub(r'[^a-zA-Z\s]+', '', text)  # Remove non-alphabetic characters
    text = text.lower()  # Convert to lowercase
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatize words
    return ' '.join(tokens)

# Apply preprocessing
train_data['cleaned_text'] = train_data['text'].apply(preprocess)
test_data['cleaned_text'] = test_data['text'].apply(preprocess)

# Vectorization
vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
x_train = vectorizer.fit_transform(train_data['cleaned_text'])
x_test = vectorizer.transform(test_data['cleaned_text'])

# Define target variables
y_train = train_data['sentiment']
y_test = test_data['sentiment']

# Handle class imbalance using SMOTE
smote = SMOTE()
x_train_balanced, y_train_balanced = smote.fit_resample(x_train, y_train)

# Define the logistic regression model
lr_model = LogisticRegression(max_iter=1000)

# Parameter tuning with GridSearchCV
param_grid = {
    'C': [0.1, 1, 10],
    'solver': ['liblinear', 'lbfgs'],
    'class_weight': ['balanced', None]
}
grid_search = GridSearchCV(lr_model, param_grid, cv=3, scoring='accuracy', verbose=1, n_jobs=-1)
grid_search.fit(x_train_balanced, y_train_balanced)

# Get the best model
best_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)

# Predict and evaluate
y_pred = best_model.predict(x_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Neutral', 'Positive']))
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# Plot the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['Negative', 'Neutral', 'Positive'])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# Apply LDA to extract topics
lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda_topics = lda.fit_transform(x_train)

# Display the top words in each topic
feature_names = vectorizer.get_feature_names_out()
n_top_words = 10
for topic_idx, topic in enumerate(lda.components_):
    print(f"Topic #{topic_idx}:")
    print(" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))

# KMeans Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(x_train)

# Add cluster labels to the training data
train_data['cluster'] = clusters

# Plot sentiment distribution within clusters
plt.figure(figsize=(10, 6))
train_data.groupby(['cluster', 'sentiment']).size().unstack().plot(kind='bar', stacked=True)
plt.title('Sentiment Distribution Across Clusters')
plt.xlabel('Cluster')
plt.ylabel('Frequency')
plt.show()

import pickle

# Paths to save pickle files
model_path = "E:\\ML_LAB_ASSIGNMENT\\best_model.pkl"
vectorizer_path = "E:\\ML_LAB_ASSIGNMENT\\vectorizer.pkl"

# Save the vectorizer
with open(vectorizer_path, 'wb') as f:
    pickle.dump(vectorizer, f)
print(f"Vectorizer saved to {vectorizer_path}")

# Save the trained model
with open(model_path, 'wb') as f:
    pickle.dump(best_model, f)
print(f"Model saved to {model_path}")

# To load the vectorizer and model later
with open(vectorizer_path, 'rb') as f:
    loaded_vectorizer = pickle.load(f)

with open(model_path, 'rb') as f:
    loaded_model = pickle.load(f)

print("Vectorizer and model loaded successfully!")