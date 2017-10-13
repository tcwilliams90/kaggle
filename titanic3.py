# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 20:15:33 2017

@author: tcwilliams
"""

# Import libraries
import os, math
import numpy as np
from random import random, randint
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns

# Modeling Algorithms
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import cross_val_score

# Modeling Helpers
from sklearn.preprocessing import Imputer , Normalizer , scale
from sklearn.cross_validation import train_test_split , StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectFromModel


# Configure visualizations
%matplotlib inline
mpl.style.use( 'ggplot' )
sns.set_style( 'white' )
pylab.rcParams[ 'figure.figsize' ] = 8 , 6

#  used guidance from https://www.kaggle.com/helgejo/an-interactive-data-science-tutorial

# Helge's helper functions

def plot_histograms( df , variables , n_rows , n_cols ):
    fig = plt.figure( figsize = ( 16 , 12 ) )
    for i, var_name in enumerate( variables ):
        ax=fig.add_subplot( n_rows , n_cols , i+1 )
        df[ var_name ].hist( bins=10 , ax=ax )
        ax.set_title( 'Skew: ' + str( round( float( df[ var_name ].skew() ) , ) ) ) # + ' ' + var_name ) #var_name+" Distribution")
        ax.set_xticklabels( [] , visible=False )
        ax.set_yticklabels( [] , visible=False )
    fig.tight_layout()  # Improves appearance a bit.
    plt.show()

def plot_distribution( df , var , target , **kwargs ):
    row = kwargs.get( 'row' , None )
    col = kwargs.get( 'col' , None )
    facet = sns.FacetGrid( df , hue=target , aspect=4 , row = row , col = col )
    facet.map( sns.kdeplot , var , shade= True )
    facet.set( xlim=( 0 , df[ var ].max() ) )
    facet.add_legend()

def plot_categories( df , cat , target , **kwargs ):
    row = kwargs.get( 'row' , None )
    col = kwargs.get( 'col' , None )
    facet = sns.FacetGrid( df , row = row , col = col )
    facet.map( sns.barplot , cat , target )
    facet.add_legend()

def plot_correlation_map( df ):
    corr = titanic.corr()
    _ , ax = plt.subplots( figsize =( 12 , 10 ) )
    cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )
    _ = sns.heatmap(
        corr, 
        cmap = cmap,
        square=True, 
        cbar_kws={ 'shrink' : .9 }, 
        ax=ax, 
        annot = True, 
        annot_kws = { 'fontsize' : 12 }
    )

def describe_more( df ):
    var = [] ; l = [] ; t = []
    for x in df:
        var.append( x )
        l.append( len( pd.value_counts( df[ x ] ) ) )
        t.append( df[ x ].dtypes )
    levels = pd.DataFrame( { 'Variable' : var , 'Levels' : l , 'Datatype' : t } )
    levels.sort_values( by = 'Levels' , inplace = True )
    return levels

def plot_variable_importance( X , y ):
    tree = DecisionTreeClassifier( random_state = 99 )
    tree.fit( X , y )
    plot_model_var_imp( tree , X , y )
    
def plot_model_var_imp( model , X , y ):
    imp = pd.DataFrame( 
        model.feature_importances_  , 
        columns = [ 'Importance' ] , 
        index = X.columns 
    )
    imp = imp.sort_values( [ 'Importance' ] , ascending = True )
    imp[ : 10 ].plot( kind = 'barh' )
    print (model.score( X , y ))
    

# Function to replace missing ages from Ahmed BESBES


def process_age():
    
    global full_set
    
    # a function that fills the missing values of the Age variable
    
    def fillAges_TW(row, grouped_median):
        return grouped_median.loc[row['Sex'], row['Pclass'], row['Title']]['Age']
           
    full_set.loc[0:890, 'Age'] = full_set.head(891).apply(lambda r : fillAges_TW(r, grouped_median_train) if np.isnan(r['Age']) 
                                                      else r['Age'], axis=1)
    
    full_set.loc[891:, 'Age'] = full_set.iloc[891:].apply(lambda r : fillAges_TW(r, grouped_median_test) if np.isnan(r['Age']) 
                                                      else r['Age'], axis=1)

def bin_age(df):
    bins = (-1, 0, 6, 12, 18, 25, 35, 60, 120)
    group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Young Adult', 'Adult', 'Middle Age', 'Senior']
    categories = pd.cut(df.Age, bins, labels=group_names)
    df.Age = categories
    return df


def process_fare():
    
    global full_set
    
    # a function that fills the missing values of the Age variable
    
    def fillFares(row, grouped_fare_median):
        return grouped_fare_median.loc[row['Pclass']]['Fare']
           
    full_set.loc[0:890, 'Fare'] = full_set.head(891).apply(lambda r : fillFares(r, grouped_fare_median_train) if np.isnan(r['Fare']) or r['Fare'] == 0
                                                      else r['Fare'], axis=1)
    
    full_set.loc[891:, 'Fare'] = full_set.iloc[891:].apply(lambda r : fillFares(r, grouped_fare_median_test) if np.isnan(r['Fare']) or r['Fare'] == 0
                                                      else r['Fare'], axis=1)
#    full_set.loc[0:890, 'Fare'].fillna(full_set.head(891).Fare.mean(),inplace=True)
    
#    full_set.loc[891:, 'Fare'].fillna(full_set.iloc[891:].Fare.mean(),inplace=True)
        


def process_multi_cabin_fares():
    
    global full_set
    
    # a function that fills the missing values of the Age variable
    
    def fixMultiCabinFares(row):
        fare = row['Fare']
        cabins = row['Cabin'].split(" ")
        if len(cabins) > 1:
            fare = fare / len(cabins)
        return fare
           
    full_set.loc[0:890, 'Fare'] = full_set.head(891).apply(fixMultiCabinFares, axis=1)
    
    full_set.loc[891:, 'Fare'] = full_set.iloc[891:].apply(fixMultiCabinFares, axis=1)

def bin_fare(df):
    bins = (-1, 0, 8, 15, 31, 1000)
    group_names = ['Unknown', 'fare_qrt_1', 'fare_qrt_2', 'fare_qrt_3', 'fare_qrt_4']
    categories = pd.cut(df.Fare, bins, labels=group_names)
    df.Fare = categories
    return df

    
def get_ticket_prefix_and_number(ticket):
    split = ticket.rsplit(" ", 1) 
    if len(split) > 1:
        tp = split[0]
        tn = split[1]
    else:
        if str.isnumeric(split[0]):
            tp = "N/A"
            tn = split[0]
        else:  # no ticket number, probably employee
            tp = split[0]
            tn = 0
    return [tp, tn]

def get_ticket_prefix(tkt):
    tkt = tkt.replace('.', '')
    tkt = tkt.replace('/', '')
    tkt = tkt.split()
    tkt = map(lambda t: t.strip(), tkt)
    tkt = filter(lambda t: not t.isdigit(), tkt)
    l_tkt = list(tkt)   # filter in Python 3 returns iterator, not list as in Py 2
    if len(l_tkt) > 0:
        return l_tkt[0]
    else:
        return 'XXX'

def recover_original_data(trn_file):
    orig_train = pd.read_csv(trn_file)
    targets = orig_train.Survived
    train = orig_train.head(891)
    test = orig_train.iloc[891:]
    
    return train, test, targets


def compute_score(cl, X, y, scoring='accuracy'):
    xv = cross_val_score(cl, X, y, cv=5, scoring=scoring)
    return np.mean(xv)
    
def dummy_age_and_fare(full_set):
    age_dummies = pd.get_dummies(full_set.Age, prefix = 'Age')
    full_set = pd.concat([full_set, age_dummies], axis=1)
    full_set.drop('Age',axis=1,inplace=True)
    
    fare_dummies = pd.get_dummies(full_set.Fare, prefix = 'Fare')
    full_set = pd.concat([full_set, fare_dummies], axis=1)
    full_set.drop('Fare',axis=1,inplace=True)



# get titanic data
    
skydrive_path = "C:/Users/tcwilliams/SkyDrive/Documents"
onedrive_path = "C:/Users/tcwilliams/OneDrive/Documents"

if os.path.isdir(skydrive_path):
    file_path = skydrive_path +  "/tcw/datasci/kaggle/titanic/"
else:
    file_path = onedrive_path +  "/tcw/datasci/kaggle/titanic/"

train_file = file_path + "train.csv"
test_file =  file_path + "test.csv"
output_file = file_path + "results.csv"

train = pd.read_csv(train_file)
test = pd.read_csv(test_file)

titanic = train

titanic.head()

titanic.describe()

plot_correlation_map(titanic)

plot_distribution(titanic, var = 'Age', target = 'Survived', row = 'Sex')
plot_distribution(titanic, var = 'Fare', target = 'Survived', row = 'Sex')
plot_categories(titanic, cat = 'Sex', target = 'Survived')
plot_categories(titanic, cat = 'Pclass', target = 'Survived')
plot_categories(titanic, cat = 'SibSp', target = 'Survived')
plot_categories(titanic, cat = 'Parch', target = 'Survived')

# Create combined data set for feature extraction.  Remove the Survived values from the training data before joining

targets = train.Survived   # save the 'y' values from the training set
train.drop('Survived', 1, inplace = True)

full_set = train.append(test, ignore_index = True)

full_set.describe()

print('datasets:', 'full_set:', full_set.shape, 'titanic:', titanic.shape)


full_set['Title'] = full_set['Name'].map(lambda name: name.split( ',')[1].split('.')[0].strip())


Title_Dictionary = {
                    "Capt":       "Official",
                    "Col":        "Official",
                    "Major":      "Official",
                    "Jonkheer":   "Royalty",
                    "Don":        "Royalty",
                    "Sir" :       "Royalty",
                    "Dr":         "Official",
                    "Rev":        "Official",
                    "the Countess":"Royalty",
                    "Dona":       "Royalty",
                    "Mme":        "Mrs",
                    "Mlle":       "Miss",
                    "Ms":         "Mrs",
                    "Mr" :        "Mr",
                    "Mrs" :       "Mrs",
                    "Miss" :      "Miss",
                    "Master" :    "Master",
                    "Lady" :      "Royalty"

                    }
# fix Titles
full_set['Title'] = full_set.Title.map(Title_Dictionary)


# Separate ticket numbers from any prefixes

#ticket_split = pd.DataFrame()

#ticket_split = full_set['Ticket'].map(get_ticket_prefix_and_number )

#full_set['TicketPrefix'] = ticket_split.map(lambda t: str.strip(t[0]))

full_set['TicketPrefix'] = full_set['Ticket'].map(get_ticket_prefix)
tp_dummies = pd.get_dummies(full_set.TicketPrefix, prefix = 'TktPrfx')
full_set = pd.concat([full_set, tp_dummies], axis=1)
full_set.drop('TicketPrefix',axis=1,inplace=True)

#full_set['TicketNumber'] = ticket_split.map(lambda t: t[1])

# Replace missing ages per Ahmed BESBES

grouped_train = full_set.head(891).groupby(['Sex', 'Pclass', 'Title'])
grouped_median_train = grouped_train.median()

grouped_test = full_set.iloc[891:].groupby(['Sex', 'Pclass', 'Title'])
grouped_median_test= grouped_test.median()

grouped_median_train
grouped_median_test

process_age()

bin_age(full_set)

# set up age groups
# age categorization doesn't seem to add any accuracy 10/12
#full_set.loc[full_set['Age'] <= 16, 'Age'] = 0
#full_set.loc[(full_set['Age'] > 16) & (full_set['Age'] <= 32), 'Age'] = 1
#full_set.loc[(full_set['Age'] > 32) & (full_set['Age'] <= 48), 'Age'] = 2
#full_set.loc[(full_set['Age'] > 48) & (full_set['Age'] <= 64), 'Age'] = 3
#full_set.loc[full_set['Age'] > 64, 'Age'] = 4


# Replace missing fares with median using Ahmed BESBES process

grouped_fare_train = full_set.head(891).groupby(['Pclass'])
grouped_fare_median_train = grouped_fare_train.mean()

grouped_fare_test = full_set.iloc[891:].groupby(['Pclass'])
grouped_fare_median_test= grouped_fare_test.median()

grouped_fare_median_train
grouped_fare_median_test

process_fare()

bin_fare(full_set)

# Dummy variables for Age and Fare
#dummy_age_and_fare(full_set)

# Add dummy variables for Titles
title_dummies  = pd.get_dummies(full_set.Title, prefix="Title")

title_dummies.head()

full_set = pd.concat([full_set, title_dummies], axis=1)
full_set.drop('Title',axis=1,inplace=True)

# map Sex to numbers
full_set['Sex'] = full_set['Sex'].map({'male': 1, 'female': 0})
#sex_dummies = pd.get_dummies(sex, prefix = 'Sex')


# replace missing Embarked values
full_set.loc[0:890, 'Embarked'].fillna('S', inplace=True)
full_set.loc[891:, 'Embarked'].fillna('S', inplace=True)

# add dummies for Embarked

embarked_dummies = pd.get_dummies(full_set.Embarked, prefix = 'Embarked')
full_set = pd.concat([full_set, embarked_dummies], axis=1)
full_set.drop('Embarked',axis=1,inplace=True)




#full_set.describe()

# Fill missing values of Fare with average (mean) fare
# Note:  some fares were paid for multiple cabins. Need to adjust for this

# full_set[ 'Fare' ] = full_set.Fare.fillna(full_set.Fare.mean())



deck = pd.DataFrame()
cabin = pd.DataFrame()

# Helge's feature extraction erroneously referred to deck as 'Cabin'

full_set['Cabin'] = full_set.Cabin.fillna('U')

cabin['Cabin'] = full_set['Cabin']
deck['Deck'] = cabin['Cabin'].map(lambda c : c[0])

# process_multi_cabin_fares()


#cabin_dummies = pd.get_dummies(cabin['Cabin'], prefix = 'Cabin')

# Initial set of models to tryfull_set.drop('Cabin',axis=1,inplace=True)
deck_dummies = pd.get_dummies(deck['Deck'], prefix = 'Deck')

full_set = pd.concat([full_set, deck_dummies], axis=1)

# add dummies for passenger class
pclass_dummies = pd.get_dummies(full_set['Pclass'], prefix="Pclass")
full_set = pd.concat([full_set, pclass_dummies], axis=1)
full_set.drop('Pclass',axis=1,inplace=True)

# full_set = pd.concat([full_set, pclass_dummies], axis=1)

#  Didn't use Helge's ticket class feature as it seemed duplicative of 'PClass'
#  Try something like Helge's FamilySize feature

family_size = pd.DataFrame()

family_size = full_set['SibSp'] + full_set['Parch'] + 1

#full_set = pd.concat([full_set, family_size], axis=1)

full_set['FamilySize'] = family_size

full_set['FamilySingle'] = full_set['FamilySize'].map(lambda s: 1 if s == 1 else 0)
full_set['FamilySmall'] = full_set['FamilySize'].map(lambda s: 1 if 2<=s<=4 else 0)
full_set['FamilyLarge'] = full_set['FamilySize'].map(lambda s: 1 if s>=5 else 0)

# drop string columns so 'fit' will work
full_set.drop('Name',axis=1,inplace=True)
full_set.drop('Ticket',axis=1,inplace=True)
full_set.drop('Cabin',axis=1,inplace=True)

# drop the PassengerID
full_set.drop('PassengerId',axis=1,inplace=True)



#age_dummies = pd.get_dummies(pd.qcut(full_set['Age'], 10), prefix="Age")

#age_dummies.head()

#fare_dummies = pd.get_dummies(pd.qcut(full_set['Fare'], 10), prefix="Fare")
    
#fare_dummies.head()

#full_set_X = pd.concat([imputed, embarked, deck, sex, full_set['Pclass'], full_set['Parch'] ], axis=1)
#full_set_X = pd.concat([deck, sex_dummies, title, family_size, full_set['Age'], full_set['Fare'], pclass_dummies ], axis=1)
#full_set_X = pd.concat([deck, sex_dummies, title, family_size, age_dummies, fare_dummies, pclass_dummies ], axis=1)
#full_set_X = pd.concat([imputed, embarked, deck, sex, full_set['Pclass'], full_set['Parch'] ], axis=1)
#full_set_X = pd.concat([deck, sex_dummies, title, family_size, full_set['Age'], full_set['Fare'], pclass_dummies ], axis=1)


#full_set_X.head()


#train_valid_X = full_set_X[0:891]
train_valid_X = full_set[0:891]
train_valid_y = targets

#test_X = full_set_X[891:]
test_X = full_set[891:]

train_X, valid_X, train_y, valid_y = train_test_split (train_valid_X, train_valid_y, train_size = .7)

print(full_set.shape, train_X.shape, valid_X.shape, train_y.shape, valid_y.shape, test_X.shape)

from pandas import Series
from matplotlib import pyplot
series = Series.from_array(train['Age'])
series.hist()
pyplot.show()


cl = RandomForestClassifier(n_estimators=100)
#cl = RandomForestClassifier(n_estimators=50, max_features='sqrt')
#cl = GaussianNB()
#cl = LogisticRegression()
#cl = KNeighborsClassifier(n_neighbors=3)
#cl = GradientBoostingClassifier()
#cl = SVC()


#import xgboost as xgb

cl.fit(train_X, train_y)

print (cl.score( train_X , train_y ) , cl.score( valid_X , valid_y ))

features = pd.DataFrame()
features['feature'] = train_X.columns
features['importance'] = cl.feature_importances_
features.sort_values(by=['importance'], ascending=True, inplace=True)
features.set_index('feature', inplace=True)

features.plot(kind='barh', figsize=(20,20))

# train, test, targets = recover_original_data(train_file)

train = full_set[0:891]
test = full_set[891:]


model = SelectFromModel(cl, prefit=True)
train_reduced = model.transform(train)
train_reduced.shape

test_reduced = model.transform(test)
test_reduced.shape



model = RandomForestClassifier(n_estimators=100)
#model = RandomForestClassifier(n_estimators=50, max_features='sqrt')
rfecv = RFECV(estimator = model, step = 1, cv = StratifiedKFold(train_y, 2), scoring='accuracy')
rfecv.fit(train_X, train_y)

print (rfecv.score( train_X , train_y ) , rfecv.score( valid_X , valid_y ))
print( "Optimal number of features : %d" % rfecv.n_features_ )

# tests from Ahmed BESBES tutorial

train_X = full_set[0:891]
train_y = targets

# Grid Search                                                                                                                       to identify best hyperparameters

run_gs = True

if run_gs:
    parameter_grid = {
                 'max_depth' : [4, 6, 8],
                 'n_estimators': [50, 10],
                 'max_features': ['sqrt', 'auto', 'log2'],
                 'min_samples_split': [3, 10],
                 'min_samples_leaf': [3, 10],
                 'bootstrap': [True, False],
                 }
    forest = RandomForestClassifier()
    cross_validation = StratifiedKFold(train_y, n_folds=5)

    grid_search = GridSearchCV(forest,
                               scoring='accuracy',
                               param_grid=parameter_grid,
                               cv=cross_validation)

    grid_search.fit(train_X, train_y)
#    grid_search.fit(train_reduced, targets)
    model = grid_search
    parameters = grid_search.best_params_

    print('Best score: {}'.format(grid_search.best_score_))
    print('Best parameters: {}'.format(grid_search.best_params_))
else: 
    parameters = {'bootstrap': False, 'min_samples_leaf': 3, 'n_estimators': 50, 
                  'min_samples_split': 10, 'max_features': 'log2', 'max_depth': 8}
    model = RandomForestClassifier(**parameters)
    model.fit(train_X, train_y)
#    model.fit(train_reduced, targets)
    compute_score(model, train_X, train_y, scoring='accuracy')
#    compute_score(model, train_reduced, targets, scoring='accuracy')

# Output the predictions

output = model.predict(test_X).astype(int)
#output = model.predict(test_reduced).astype(int)
df_output = pd.DataFrame()
aux = pd.read_csv(test_file)
df_output['PassengerId'] = aux['PassengerId']
df_output['Survived'] = output
df_output[['PassengerId', 'Survived']].to_csv(output_file, index=False)