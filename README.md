# Duplicate Bug Report Detection

During the hands-on session, we formulate duplicate bug report detection (DBRD) as a binary classification task[^1] (i.e., classify a bug report as whether duplicate or not).

[^1]: Chengnian Sun, David Lo, Xiaoyin Wang, Jing Jiang, and Siau-Cheng Khoo. 2010. [A discriminative model approach for accurate duplicate bug report retrieval](https://dl.acm.org/doi/pdf/10.1145/1806799.1806811). In Proceedings of the 32nd ACM/IEEE International Conference on Software Engineering - Volume 1 (ICSE '10). Association for Computing Machinery, New York, NY, USA, 45–54. DOI:https://doi.org/10.1145/1806799.1806811

## Learning Objective
 * Understand how duplicate bug report detection work
 * Be able to classify duplicate bug reports

## Prerequisite
 * Installation of a package manager, `conda` (https://docs.conda.io/en/latest/). Then, install the below Python libraries via `conda`:
   * pandas: https://pandas.pydata.org/
   * scikit-learn: https://scikit-learn.org/stable/

If you use M1 Mac (ARM architecture) and cannot install the conda properly, please check this article: https://blog.donggyun.com/article/4

## A Dataset for DBRD

The dataset covers the bug reports submitted to Mozilla Bugzilla. The crawled date range is from 1 Jan 2021 to 30 May 2021. It contains 15,076 pre-processed bug reports in JSON format. A bug report in the dataset looks as below:

```json
{
    "bug_id":1710189,
    "title":"Lock Icon overlaps text in connection info popup in Proton",
    "status":"RESOLVED",
    "resolution":"DUPLICATE",
    "product":"Firefox",
    "component":"Site Identity",
    "version":"Firefox 90",
    "priority":"--",
    "type":"defect",
    "description":"Created attachment 9220926... "
}
```

## Categorical Information Based DBRD 
### 1.	Clone the tutorial repo
 
```
> git clone git@github.com:handk85/dbrd-hands-on.git
> cd dbrd-hands-on 
```

### 2. Classification script	
Implementation (scripts/dbrd-rf.py)

```python
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
```

Output
```
[[1185   87]
 [ 126  110]]
              precision    recall  f1-score   support

       False       0.90      0.93      0.92      1272
        True       0.56      0.47      0.51       236

    accuracy                           0.86      1508
   macro avg       0.73      0.70      0.71      1508
weighted avg       0.85      0.86      0.85      1508

Accuracy: 0.8587533156498673
AUC-ROC: 0.69885273425008
```


## Tasks
### Task 1
Please use 100% dataset for the classification and check the performance difference.

### Task 2
Please use 10% of dataset as test dataset. Also, please use 30% of dataset as test dataset.

### Task 3
Please replace the random forest classifier with decision tree classifier.
You can import decision tree classifier as below:
```
from sklear.tree import DecisionTreeClassifier
```

You can also try other classifiers in scikit-learn (https://scikit-learn.org/stable/modules/classes.html).


## Natural Language Based DBRD
### 1. Classification script
Implementation (scripts/dbrd-mlp.py)

```python
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
# This lambda converts resolution into either True (i.e., duplicate) or False (i.e., non-duplicate)
dataset[LABEL_FIELD] = dataset[LABEL_FIELD].apply(lambda x: x == "DUPLICATE")
# Dataset MUST BE SORTED! You cannot train a model with future data in real practice.
dataset = dataset.sort_values("bug_id")

# Select 250 bug reports for each label
duplicates = dataset[dataset.resolution].sample(n=250)
non_duplicates = dataset[~dataset.resolution].sample(n=250)
dataset = pd.concat([duplicates, non_duplicates])

# Concatenate title and description
texts = dataset["title"] + "\n\n" + dataset["description"]
# Even though we use the concatenated string directly, you can adopt pre-processing techniques (e.g., stopword removal)

X = texts.values.ravel()
y = dataset[LABEL_FIELD].values.ravel()

# Split dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Use bag of words to convert texts into vectors
cv = CountVectorizer()
X_train = cv.fit_transform(X_train)
X_test = cv.transform(X_test)

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
```

Output
```
[[238  19]
 [ 33  12]]
              precision    recall  f1-score   support

       False       0.88      0.93      0.90       257
        True       0.39      0.27      0.32        45

    accuracy                           0.83       302
   macro avg       0.63      0.60      0.61       302
weighted avg       0.81      0.83      0.81       302

Accuracy: 0.8278145695364238
AUC-ROC: 0.5963683527885862
```

## Tasks
### Task 1
The above implementation does not leverage natural language pre-processing techniques (e.g., stopwords removal). Please install `nltk` via `conda`.
```
conda install nltk
```

You can modify `scripts/dbrd-mlp.py`  to remove the common stopwords in the dataset as below:

```python
... 
from nltk.corpus import stopwords

# The below lines only need to be called once on your machine.
import nltk
nltk.download('stopwords')
...

# Locate the below line in a proper position in the source code
texts=texts.apply(lambda x: " ".join([word for word in x.split() if word not in (stopwords.words('english'))]))
```

### Task 2
Please adopt stemming in `scripts/dbrd-mlp.py` by using Porter stemmer.
The expected change is as below:

```python
...
from nltk.stem.porter import PorterStemmer
...

# Locate the below lines in a proper position in the source code
stemmer = PorterStemmer()
texts=texts.apply(lambda x: " ".join([stemmer.stem(word) for word in x.split()]))
```

### Task 3
We only used small amount of the entire dataset due to the time limits. Please try the entire dataset to train the model.

