import pandas as pd
import scipy as sp
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score

# The names of properties defined in pre-processed data
CATEGORICAL_FEATURES = ["product", "component", "version", "priority", "type"]
NL_FEATURES = ["title", "description"]
LABEL_FIELD = ["resolution"]

# Load data
dataset = pd.read_json("../data/preprocessed-data-MOZILLA.json")
# This lambda converts resolution into either True (i.e., duplicate) or False (i.e., non-duplicate)
dataset[LABEL_FIELD] = dataset[LABEL_FIELD].apply(lambda x: x == "DUPLICATE")
# Dataset MUST BE SORTED! You cannot train a model with future data in real practice.
dataset = dataset.sort_values("bug_id")

# Select 250 bug reports for each label
duplicates = dataset[dataset.resolution].sample(n=250)
non_duplicates = dataset[~dataset.resolution].sample(n=250)
dataset = pd.concat([duplicates, non_duplicates])

# Vectorize the categorical features
le = LabelEncoder()
categorical_features = dataset[CATEGORICAL_FEATURES].apply(le.fit_transform)

# You can adopt pre-processing techniques here
texts = dataset["title"] + "\n\n" + dataset["description"]

# Prepare features and labels
X = pd.concat([categorical_features, texts.rename("texts")], axis=1)
y = dataset[LABEL_FIELD].values.ravel()

# Split dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Use bag of words to convert texts into vectors
cv = CountVectorizer()
X_train = sp.sparse.hstack((cv.fit_transform(X_train.texts), X_train[CATEGORICAL_FEATURES]), format='csr')
X_test = sp.sparse.hstack((cv.transform(X_test.texts), X_test[CATEGORICAL_FEATURES]), format='csr')

# standardize both train and test data set to optimize the model
scaler = StandardScaler(with_mean=False)
# fit only on training data
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

mlp = MLPClassifier()
mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_test)

# Print the results
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Accuracy: %s" % accuracy_score(y_test, y_pred))
print("AUC-ROC: %s" % roc_auc_score(y_test, y_pred))
