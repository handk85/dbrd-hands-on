import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score

# The names of properties defined in pre-processed data
CATEGORICAL_FEATURES = ["product", "component", "version", "priority", "type"]
LABEL_FIELD = ["resolution"]

# Load data
dataset = pd.read_json("../data/preprocessed-data-MOZILLA.json")
# This lambda converts resolution into either True (i.e., duplicate) or False (i.e., non-duplicate)
dataset[LABEL_FIELD] = dataset[LABEL_FIELD].apply(lambda x: x == "DUPLICATE")
# Dataset MUST BE SORTED! You cannot train a model with future data in real practice.
dataset = dataset.sort_values("bug_id")
# Sample the dataset. We will use only 50% of the randomly selected dataset.
dataset = dataset.sample(frac=0.5)

# Since classification algorithm cannot take string values, transform the string values into numeric values
le = LabelEncoder()
features = dataset[CATEGORICAL_FEATURES]
X = features.apply(le.fit_transform)

y = dataset[LABEL_FIELD].values.ravel()

# Split dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# standardize both train and test data set to optimize the model
scaler = StandardScaler()
# fit only on training data
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# Print the results
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Accuracy: %s" % accuracy_score(y_test, y_pred))
print("AUC-ROC: %s" % roc_auc_score(y_test, y_pred))
