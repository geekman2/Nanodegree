# Import libraries
import numpy as np
import pandas as pd
import os
from sklearn.metrics import make_scorer
from matplotlib import pyplot as plt

# Read student data
student_data = pd.read_csv("student-data.csv")
student_data.reindex(np.random.permutation(student_data.index))
print "Student data read successfully!"
# Note: The last column 'passed' is the target/label, all other are feature columns

# TODO: Compute desired values - replace each '?' with an appropriate expression/function call
print type(student_data)
n_students = float(len(student_data))
n_features = len(student_data.columns)
n_passed = len(student_data[student_data.passed == 'yes'])
n_failed = len(student_data[student_data.passed == 'no'])
grad_rate = n_passed / n_students * 100
print "Total number of students: {}".format(n_students)
print "Number of students who passed: {}".format(n_passed)
print "Number of students who failed: {}".format(n_failed)
print "Number of features: {}".format(n_features)
print "Graduation rate of the class: {:.2f}%".format(grad_rate)

# Extract feature (X) and target (y) columns
feature_cols = list(student_data.columns[:-1])  # all columns but last are features
target_col = student_data.columns[-1]  # last column is the target/label
print "Feature column(s):-\n{}".format(feature_cols)
print "Target column: {}".format(target_col)

X_all = student_data[feature_cols]  # feature values for all students
y_all = student_data[target_col]  # corresponding targets/labels
print "\nFeature values:-"
print X_all.head()  # print the first 5 rows


# Preprocess feature columns
def preprocess_features(X):
    outX = pd.DataFrame(index=X.index)  # output dataframe, initially empty

    # Check each column
    for col, col_data in X.iteritems():
        # If data type is non-numeric, try to replace all yes/no values with 1/0
        if col_data.dtype == object:
            col_data = col_data.replace(['yes', 'no'], [1, 0])
        # Note: This should change the data type for yes/no columns to int

        # If still non-numeric, convert to one or more dummy variables
        if col_data.dtype == object:
            col_data = pd.get_dummies(col_data, prefix=col)  # e.g. 'school' => 'school_GP', 'school_MS'

        outX = outX.join(col_data)  # collect column(s) in output dataframe

    return outX


X_all = preprocess_features(X_all)
y_all = y_all == "yes"
y_all = y_all.astype(int)
#print "Processed feature columns ({}):-\n{}".format(len(X_all.columns), list(X_all.columns))

def train_test_split(num_train):
    num_all = student_data.shape[0]  # same as len(student_data)
    num_test = num_all - num_train

    # TODO: Then, select features (X) and corresponding labels (y) for the training and test sets
    # Note: Shuffle the data or randomly select samples to avoid any bias due to ordering in the dataset
    X_train = X_all[0:num_train] #X_all.sample(frac = frac_train)
    y_train = y_all[0:num_train] #y_all.sample(frac = frac_train)#
    X_test = X_all.sample(frac = 0.3)#X_all[300:]
    y_test = y_all.sample(frac = 0.3)#y_all[300:]
    #print "Training set: {} samples".format(X_train.shape[0])
    #print "Test set: {} samples".format(X_test.shape[0])
    return X_train,y_train,X_test,y_test




# Train a model
import time

def train_classifier(clf, X_train, y_train):
    print "Training {}...".format(clf.__class__.__name__)
    start = time.time()
    clf.fit(X_train, y_train)
    end = time.time()
    print "Training time (secs): {:.3f}".format(end - start)


# TODO: Choose a model, import it and instantiate an object
from sklearn import tree, svm, naive_bayes, ensemble, neighbors

dtc = tree.DecisionTreeClassifier()
svc = svm.SVC()
nbc = naive_bayes.GaussianNB()
knn = neighbors.KNeighborsClassifier()
rfc = ensemble.RandomForestClassifier()
adc = ensemble.AdaBoostClassifier()

models = [dtc,svc,nbc,knn,rfc,adc]


# Fit model to training data


# note: using entire training set here

# Predict on training set and compute F1 score
from sklearn.metrics import f1_score

def predict_labels(clf, features, target):
    #print "Predicting labels using {}...".format(clf.__class__.__name__)
    start = time.time()
    y_pred = clf.predict(features)
    end = time.time()
    print "Prediction time (secs): {:.3f}".format(end - start)
    return f1_score(target.values, y_pred)

def train_predict(clf, X_train, y_train, X_test, y_test):
    print "------------------------------------------"
    print "Training set size: {}".format(len(X_train))
    train_classifier(clf, X_train, y_train)
    f1_train = predict_labels(clf, X_train, y_train)
    print "F1 score for training set: {}".format(f1_train)
    f1_test = predict_labels(clf, X_test, y_test)
    print "F1 score for test set: {}".format(f1_test)
    return f1_train,f1_test,len(X_train)


#for clf in models:
#    for x in [100,200,300]:
#        X_train,y_train,X_test,y_test = train_test_split(x)
#        train_predict(clf,X_train,y_train,X_test,y_test)


X_train,y_train,X_test,y_test = train_test_split(300)

from sklearn.metrics import make_scorer
from sklearn import grid_search
from sklearn import svm
from sklearn.metrics import make_scorer
# TODO: Import 'GridSearchCV' and 'make_scorer'

# TODO: Create the parameters list you wish to tune
parameters = [{"kernel":["poly"],
                "degree":[1,2,3,4,5],
                "C":[1,10,100,1000],
             }]

# TODO: Initialize the classifier
clf = svm.SVC()

# TODO: Make an f1 scoring function using 'make_scorer'
f1_scorer = make_scorer("f1",greater_is_better=False)

# TODO: Perform grid search on the classifier using the f1_scorer as the scoring method
grid_obj = grid_search.GridSearchCV(clf,parameters,scoring=f1_scorer,cv=4)

# TODO: Fit the grid search object to the training data and find the optimal parameters
grid_obj = grid_obj.fit(X_train,y_train)

# Get the estimator
clf = grid_obj.best_estimator_

# Report the final F1 score for training and testing after parameter tuning
print "Tuned model has a training F1 score of {:.4f}.".format(predict_labels(clf, X_train, y_train))
print "Tuned model has a testing F1 score of {:.4f}.".format(predict_labels(clf, X_test, y_test))
