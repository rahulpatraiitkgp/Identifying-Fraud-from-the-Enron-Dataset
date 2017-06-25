#!/usr/bin/python

import sys
import pickle
import numpy
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data

### Task 1: Select Features:
### features_list contains all the features available:
features_list = ['poi', 'salary',  'bonus', 'expenses', 'fraction_from_this_person_to_poi','fraction_from_poi_to_this_person', \
                 'shared_receipt_with_poi', 'exercised_stock_options','long_term_incentive','deferred_income',\
                 'total_stock_value', 'restricted_stock','total_payments']

### Load the dictionary containing the dataset:
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

### Task 2: Remove outliers:
### I removed the "TOTAL" entry as it is clearly an outlier as described in the documentation
### I found some features with negative values (I'm not a financial expert, but looks to me it is an error)
### For features in features_list[1:] I changed values < 0 to "NaN"
data_dict.pop( "TOTAL", 0 )
for element in data_dict:
    for feature in features_list[1:]:
        if feature == 'fraction_from_this_person_to_poi' or feature == 'fraction_from_poi_to_this_person': continue
        if data_dict[element][feature] < 0: data_dict[element][feature] = "NaN"

### Task 3: Create 2 new Features 'fraction_from_this_person_to_poi' and 'fraction_from_poi_to_this_person':
### I have generated these features with values different than "NaN" with the condition for each element: 'from_this_person_to_poi', 'from_messages' and 'from_poi_to_this_person' are different than "NaN"
for element in data_dict:
    if data_dict[element][ 'from_this_person_to_poi'] == "NaN" or data_dict[element][ 'from_messages'] == "NaN":
        data_dict[element]['fraction_from_this_person_to_poi'] = "NaN"
    else:
        data_dict[element]['fraction_from_this_person_to_poi'] = float(data_dict[element][ 'from_this_person_to_poi'])/float(data_dict[element][ 'from_messages'])    
    if data_dict[element][ 'from_poi_to_this_person'] == "NaN" or data_dict[element][ 'to_messages'] == "NaN":
        data_dict[element]['fraction_from_poi_to_this_person'] = "NaN"
    else:
        data_dict[element]['fraction_from_poi_to_this_person'] = float(data_dict[element][ 'from_poi_to_this_person'])/float(data_dict[element][ 'to_messages'])  

### Creation of labels,features: 
data = featureFormat(data_dict, features_list, sort_keys = False, remove_all_zeroes=True)
labels, features = targetFeatureSplit(data)
### I have created features_names and names_information to store the information:
### features_names is a list of persons in the same order it appears in features list, that is: features[j] are the features of person in features_names[j]
### names_information is a dictionary for each person that contains the number features different than 0 for each particular person
features_names = []
names_information = {}
for i,item in enumerate(features):
    for name in data_dict:
        matches = 0
        total_matches = len(item) - len([x for x in item if x == 0])
        for j,element in enumerate(item):
            if element != 0 and data_dict[name][features_list[j + 1]] == element:
                matches +=1
        if matches == total_matches: 
            print i,name,matches
            features_names.append(name)
            names_information[name] = matches
            break
 
### Feature Scaling: MinMaxScaler
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
features_minmax = min_max_scaler.fit_transform(features)

### Feature Selection: SelectKBest with K = 5
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
best_features = SelectKBest(chi2, k=5).fit_transform(features_minmax, labels)
### features_selected is a list that contains the position of the best features, selected just previously, accordingly to the features_list:
features_selected = []
for i,item in enumerate(best_features):
    if len([x for x in item if x == 0]) == 0:  
        for k,element in enumerate(features_minmax[i]):
            for selected_item in item:
                if selected_item == element: features_selected.append(k + 1)
        break
### Time to create my dataset:
my_dataset = {}
### With the information in features_selected, I will get the names of the best features in features_selected_name 
features_selected_name = []
first = True
for i,item in enumerate(best_features):
    ### IMPORTANT: Since 0 and "NaN" are kind of similar and doesn't bring much information, those persons with more than 3 0's among their best features, are not included in my_dataset 
    if len(item) - len([x for x in item if x == 0]) <=2 : continue
    my_dataset[features_names[i]] = {}
    my_dataset[features_names[i]]['poi'] = labels[i]
    for j,element in enumerate(item):
        my_dataset[features_names[i]][features_list[features_selected[j]]] = element
        if first:
            features_selected_name.append(features_list[features_selected[j]])
    first = False    

# So this is the feature_list used as input for the models below:
features_list = ['poi'] + features_selected_name
print features_list
#print features_selected





### Extract features and labels from my_dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.naive_bayes import GaussianNB
"""
clf = AdaBoostClassifier(DecisionTreeClassifier(criterion='entropy', min_samples_split=5),
                         algorithm="SAMME",
                         n_estimators=100)
clf =  GaussianNB()
                         
test_classifier(clf, my_dataset, features_list, folds = 3000)

"""

max_depth_input = [None]#[100,200,300,400,None]
min_samples_split_input= [5]#[2,3,4,5,6,10,20]
# finally the configuration used is: criterion='entropy', min_samples_split=5
for maxdepth in max_depth_input:
    for samples in min_samples_split_input:
        print samples,maxdepth
        clf = DecisionTreeClassifier(criterion='entropy', min_samples_split=samples,max_depth = maxdepth) 
        test_classifier(clf, my_dataset, features_list,folds = 1000)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
from sklearn import cross_validation
from sklearn import metrics
from sklearn.cross_validation import StratifiedShuffleSplit

for score in ['recall', 'precision']:
    scores = cross_validation.cross_val_score(clf, features, labels, cv=StratifiedShuffleSplit(labels,n_iter = 500), scoring='recall')
    print score + " : ", numpy.average(scores)


### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.

dump_classifier_and_data(clf, my_dataset, features_list)
