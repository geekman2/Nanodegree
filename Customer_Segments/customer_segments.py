from sklearn import tree
from sklearn.cross_validation import train_test_split
import numpy as np
import pandas as pd
from IPython.display import display # Allows the use of display() for DataFrames

# Load the wholesale customers dataset
try:
    data = pd.read_csv("customers.csv")
    data.drop(['Region', 'Channel'], axis = 1, inplace = True)
    print "Wholesale customers dataset has {} samples with {} features each.".format(*data.shape)
except:
    print "Dataset could not be loaded. Is the dataset missing?"
# TODO: Make a copy of the DataFrame, using the 'drop' function to drop the given feature
new_data = data.drop("Grocery",axis=1)

# TODO: Split the data into training and testing sets using the given feature as the target
X_train, X_test, y_train, y_test = train_test_split(new_data,
                                    data.Grocery,
                                    test_size=0.25,
                                    random_state=1791)

# TODO: Create a decision tree regressor and fit it to the training set
regressor = tree.DecisionTreeRegressor(random_state=1791)
regressor.fit(X_train,y_train)

# TODO: Report the score of the prediction using the testing set
score = regressor.score(X_test,y_test)
print score
# TODO: Scale the data using the natural logarithm
log_data = np.log(data)

# Produce a scatter matrix for each pair of newly-transformed features
pd.scatter_matrix(log_data, alpha = 0.3, figsize = (14,8), diagonal = 'kde')
# For each feature find the data points with extreme high or low values
for feature in log_data.keys():
    
    # TODO: Calculate Q1 (25th percentile of the data) for the given feature
    Q1 = np.percentile(log_data[feature],25)
#    print Q1
    
    # TODO: Calculate Q3 (75th percentile of the data) for the given feature
    Q3 = np.percentile(log_data[feature],75)
#    print Q3
    
    # TODO: Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)
    step = (Q3-Q1)*1.5
#    print step
    
#    print "Q1-step",Q1-step
#    print "Q3+step",Q3+step
    
    # Display the outliers
#    print "Data points considered outliers for the feature '{}':".format(feature)
    #display(log_data[~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step))])
    
# OPTIONAL: Select the indices for data points you wish to remove 
# step = np.percentile(log_data[feature],75)-np.percentile(log_data[feature],25)
outliers  = [item for sublist in[list(log_data[feature].index[~((log_data[feature] >= np.percentile(log_data[feature],25) - (np.percentile(log_data[feature],75)-np.percentile(log_data[feature],25))) & (log_data[feature] <= np.percentile(log_data[feature],75) + (np.percentile(log_data[feature],75)-np.percentile(log_data[feature],25))*1.5))]) for feature in log_data.keys()] for item in sublist]

print len(outliers)
outliers_intersection = set([x for x in outliers if outliers.count(x) > 1])
print outliers
print outliers_intersection

# Remove the outliers, if any were specified
good_data = log_data.drop(log_data.index[outliers]).reset_index(drop = True)
#print good_data