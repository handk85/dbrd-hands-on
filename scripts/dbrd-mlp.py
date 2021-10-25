import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score

# The names of properties defined in pre-processed data
NL_FEATURES = ["title", "description"]
LABEL_FIELD = ["resolution"]

# Load data
dataset = pd.read_json("../data/preprocessed-data-MOZILLA.json")
# Dataset MUST BE SORTED! You cannot train a model with future data in real practice.
dataset = dataset.sort_values("bug_id")
# Since it takes too long time, sample 10% of dataset for the tutorial
dataset = dataset.sample(frac=0.1)

# Concatenate title and description
texts = dataset["title"] + "\n\n" + dataset["description"]
# Even though we use the concatenated string directly, you can adopt pre-processing techniques (e.g., stopword removal)

# Use bag of words to convert texts into vectors
cv = CountVectorizer()
X = cv.fit_transform(texts).toarray()

resolution = dataset[LABEL_FIELD]
# This lambda converts resolution into either True (i.e., duplicate) or False (i.e., non-duplicate)
y = resolution.apply(lambda x: x == "DUPLICATE").values.ravel()

# Split dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# standardize both train and test data set to optimize the model
scaler = StandardScaler()
# fit only on training data
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

mlp = MLPClassifier()
mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_test)

# Print the results
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Accuracy: %s" % accuracy_score(y_test, y_pred))
print("AUC-ROC: %s" % roc_auc_score(y_test, y_pred))
