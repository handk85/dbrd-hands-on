# Duplicate Bug Report Detection

During the hands-on session, we formulate duplicate bug report detection (DBRD) as a binary classification task (i.e., classify a bug report as whether duplicate or not). 

## Learning Objective
 * Understand how duplicate bug report detection work
 * Be able to classify duplicate bug reports

## Prerequisite
 * Installation of a package manager, conda[^1] (https://docs.conda.io/en/latest/). Then, install the below Python libraries via conda:
   * pandas: https://pandas.pydata.org/
   * scikit-learn: https://scikit-learn.org/stable/

[^1]: If you use M1 Mac (ARM architecture) and cannot install the conda properly, please check this article: https://blog.donggyun.com/article/4

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

## DBRD
### 1.	Clone the tutorial repo
 
```
> git clone git@github.com:handk85/dbrd-hands-on.git
> cd msr-tutorial
```

### 2.	Categorical Information based DBRD
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
# Dataset MUST BE SORTED! You cannot train a model with future data in real practice.
dataset = dataset.sort_values("bug_id")
# Sample the dataset. We will use only 50% of the dataset.
dataset = dataset.sample(frac=0.5)

# Since classification algorithm cannot take string values, transform the string values into numeric values
le = LabelEncoder()
features = dataset[CATEGORICAL_FEATURES]
X = features.apply(le.fit_transform)

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
Please use the entire dataset for the classification and check the performance difference. Also, please try changing the train and test set ratio. 

### Task 2
Please replace the random forest classifier with another classifier. You can find classifiers implemented in scikit-learn API document (https://scikit-learn.org/stable/modules/classes.html).


