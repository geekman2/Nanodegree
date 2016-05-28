# Import libraries
import numpy as np
import pandas as pd

# Read student data
student_data = pd.read_csv("student-data.csv")
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
#print "Processed feature columns ({}):-\n{}".format(len(X_all.columns), list(X_all.columns))

# First, decide how many training vs test samples you want
num_all = student_data.shape[0]  # same as len(student_data)
num_train = 300  # about 75% of the data
num_test = num_all - num_train

# TODO: Then, select features (X) and corresponding labels (y) for the training and test sets
# Note: Shuffle the data or randomly select samples to avoid any bias due to ordering in the dataset
student_data.reindex(np.random.permutation(student_data.index))
X_train = X_all[0:num_train]
y_train = y_all[0:num_train]
X_test = X_all[num_train:]
y_test = y_all[num_train:]
print "Training set: {} samples".format(X_train.shape[0])
print "Test set: {} samples".format(X_test.shape[0])
# Note: If you need a validation set, extract it from within training data


# Train a model
import time
from sklearn.metrics import make_scorer, f1_score
from sklearn import grid_search


def train_classifier(clf, X_train, y_train,parameters):
    print "Training {}...".format(clf.__class__.__name__)
    start = time.time()
    # clf.fit(X_train, y_train)
    # Set up the parameters we wish to tune

    f1_scorer = make_scorer(f1_score, pos_label="yes")

    # Make the GridSearchCV object
    clsf = grid_search.GridSearchCV(clf, parameters, scoring=f1_scorer)

    # Fit the learner to the data to obtain the optimal model with tuned parameters
    clsf.fit(X_train, y_train)

    end = time.time()
    return clsf
    print "Done!\nTraining time (secs): {:.3f}".format(end - start)


# TODO: Choose a model, import it and instantiate an object
from sklearn import tree, svm,naive_bayes
from sklearn.metrics import f1_score

dtc = tree.DecisionTreeClassifier()
svc = svm.SVC()
nbc = naive_bayes.GaussianNB()


# Fit model to training data
dtc = train_classifier(dtc, X_train, y_train,
                       parameters={"min_samples_split":(range(1,20))})
svc = train_classifier(svc, X_train, y_train,
                       parameters={"C":(10.0,100.0,1000.0),"degree":(range(5,20)),
                                    "gamma":(10.0,100.0,1000.0)})# note: using entire training set here
#dtc = dtc.fit(X_train,y_train)
#svc = svc.fit(X_train,y_train)
nbc = nbc.fit(X_train,y_train)

#grid_predict = clf.predict(X_test)
dtc_predict = dtc.predict(X_test)
svc_predict = svc.predict(X_test)
nbc_predict = nbc.predict(X_test)

#print "Grid Search F1 Score",f1_score(y_test,grid_predict,pos_label="yes")
print "Decision Tree F1 score ",f1_score(y_test,dtc_predict,pos_label="yes")
print dtc.best_params_
print "SVM F1 score ",f1_score(y_test,svc_predict,pos_label="yes")
print svc.best_score_
print "Naive Bayes F1 score ",f1_score(y_test,nbc_predict,pos_label="yes")
# print clf  # you can inspect the learned model by printing it
