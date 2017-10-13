# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 20:15:33 2017

@author: tcwilliams
"""

# Import libraries
import numpy as np
from random import random, randint
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns


# Import Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


# Modeling Algorithms
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

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

import tensorflow as tf

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
    
    def fillAges_BESBES(row, grouped_median):
        if row['Sex']=='female' and row['Pclass'] == 1:
            if row['Title'] == 'Miss':
                return grouped_median.loc['female', 1, 'Miss']['Age']
            elif row['Title'] == 'Mrs':
                return grouped_median.loc['female', 1, 'Mrs']['Age']
            elif row['Title'] == 'Officer':
                return grouped_median.loc['female', 1, 'Official']['Age']
            elif row['Title'] == 'Royalty':
                return grouped_median.loc['female', 1, 'Royalty']['Age']

        elif row['Sex']=='female' and row['Pclass'] == 2:
            if row['Title'] == 'Miss':
                return grouped_median.loc['female', 2, 'Miss']['Age']
            elif row['Title'] == 'Mrs':
                return grouped_median.loc['female', 2, 'Mrs']['Age']

        elif row['Sex']=='female' and row['Pclass'] == 3:
            if row['Title'] == 'Miss':
                return grouped_median.loc['female', 3, 'Miss']['Age']
            elif row['Title'] == 'Mrs':
                return grouped_median.loc['female', 3, 'Mrs']['Age']

        elif row['Sex']=='male' and row['Pclass'] == 1:
            if row['Title'] == 'Master':
                return grouped_median.loc['male', 1, 'Master']['Age']
            elif row['Title'] == 'Mr':
                return grouped_median.loc['male', 1, 'Mr']['Age']
            elif row['Title'] == 'Officer':
                return grouped_median.loc['male', 1, 'Official']['Age']
            elif row['Title'] == 'Royalty':
                return grouped_median.loc['male', 1, 'Royalty']['Age']

        elif row['Sex']=='male' and row['Pclass'] == 2:
            if row['Title'] == 'Master':
                return grouped_median.loc['male', 2, 'Master']['Age']
            elif row['Title'] == 'Mr':
                return grouped_median.loc['male', 2, 'Mr']['Age']
            elif row['Title'] == 'Officer':
                return grouped_median.loc['male', 2, 'Official']['Age']

        elif row['Sex']=='male' and row['Pclass'] == 3:
            if row['Title'] == 'Master':
                return grouped_median.loc['male', 3, 'Master']['Age']
            elif row['Title'] == 'Mr':
                return grouped_median.loc['male', 3, 'Mr']['Age']
    
    full_set.head(891).Age = full_set.head(891).apply(lambda r : fillAges_TW(r, grouped_median_train) if np.isnan(r['Age']) 
                                                      else r['Age'], axis=1)
    
    full_set.iloc[891:].Age = full_set.iloc[891:].apply(lambda r : fillAges_TW(r, grouped_median_test) if np.isnan(r['Age']) 
                                                      else r['Age'], axis=1)
    
    status('age')
    
    
# get titanic data

train_file = "C:/Users/tcwilliams/SkyDrive/Documents/tcw/datasci/kaggle/titanic/train.csv"
test_file = "C:/Users/tcwilliams/SkyDrive/Documents/tcw/datasci/kaggle/titanic/test.csv"
output_file = "C:/Users/tcwilliams/SkyDrive/Documents/tcw/datasci/kaggle/titanic/results.csv"

train = pd.read_csv(train_file)
test = pd.read_csv(test_file)

targets = train.Survived
train.drop('Survived', 1, inplace = True)

full_set = train.append(test, ignore_index = True)

titanic = full_set [:891]

print('datasets:', 'full_set:', full_set.shape, 'titanic:', titanic.shape)

titanic.head()

titanic.describe()

plot_correlation_map(titanic)

plot_distribution(titanic, var = 'Age', target = 'Survived', row = 'Sex')
plot_distribution(titanic, var = 'Fare', target = 'Survived', row = 'Sex')
plot_categories(titanic, cat = 'Sex', target = 'Survived')
plot_categories(titanic, cat = 'Pclass', target = 'Survived')
plot_categories(titanic, cat = 'SibSp', target = 'Survived')
plot_categories(titanic, cat = 'Parch', target = 'Survived')

sex = pd.Series(np.where(full_set.Sex == 'male', 1, 0), name = 'Sex')

embarked = pd.get_dummies(full_set.Embarked, prefix = 'Embarked')

#  Create imputed values

imputed = pd.DataFrame()

# Replace missing ages per Ahmed BESBES

# Fill missing values of Age with *median* value

imputed[ 'Age' ] = full_set.Age.fillna(full_set.Age.median())

# Fill missing values of Fare with average (mean) fare
# Note:  some fares were paid for multiple cabins. Need to adjust for this

imputed[ 'Fare' ] = full_set.Fare.fillna(full_set.Fare.mean())

imputed.head()

title = pd.DataFrame()
title ['Title'] = full_set['Name'].map(lambda name: name.split( ',')[1].split('.')[0].strip())


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

title ['Title'] = title.Title.map(Title_Dictionary)
title  = pd.get_dummies(title.Title)

title.head()

deck = pd.DataFrame()
cabin = pd.DataFrame()

# Helge's feature extraction erroneously referred to deck as 'Cabin'

cabin['Cabin'] = full_set.Cabin.fillna('U')
cabin = pd.get_dummies(cabin['Cabin'], prefix = 'Cabin')

deck['Deck'] = cabin['Cabin'].map(lambda c : c[0])

deck = pd.get_dummies(deck['Deck'], prefix = 'Deck')


#  Didn't use Helge's ticket class feature as it seemed duplicative of 'PClass'
#  Didn't use Helge's FamilySize feature

#full_set_X = pd.concat([imputed, embarked, deck, sex, full_set['Pclass'], full_set['Parch'] ], axis=1)
full_set_X = pd.concat([imputed, deck, sex, title, full_set['Pclass'], full_set['Parch'] ], axis=1)


full_set_X.head()


train_valid_X = full_set_X[0:891]
train_valid_y = titanic.Survived

test_X = full_set_X[891:]

train_X, valid_X, train_y, valid_y = train_test_split (train_valid_X, train_valid_y, train_size = .7)

print(full_set_X.shape, train_X.shape, valid_X.shape, train_y.shape, valid_y.shape, test_X.shape)

model = RandomForestClassifier(n_estimators=100)
model = GaussianNB()
model = LogisticRegression()
model = KNeighborsClassifier(n_neighbors=3)
model = GradientBoostingClassifier()
model = SVC()

model.fit(train_X, train_y)

print (model.score( train_X , train_y ) , model.score( valid_X , valid_y ))

features = pd.DataFrame()
features['feature'] = train_X.columns
features['importance'] = model.feature_importances_
features.sort_values(by=['importance'], ascending=True, inplace=True)
features.set_index('feature', inplace=True)

features.plot(kind='barh', figsize=(20,20))


model = RandomForestClassifier(n_estimators=100)
rfecv = RFECV(estimator = model, step = 1, cv = StratifiedKFold(train_y, 2), scoring='accuracy')
rfecv.fit(train_X, train_y)

print (rfecv.score( train_X , train_y ) , rfecv.score( valid_X , valid_y ))
print( "Optimal number of features : %d" % rfecv.n_features_ )

# tests from Ahmed BESBES tutorial

grouped_train = full_set.head(891).groupby('Sex','Pclass', 'Title')

run_gs = False

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
    model = grid_search
    parameters = grid_search.best_params_

    print('Best score: {}'.format(grid_search.best_score_))
    print('Best parameters: {}'.format(grid_search.best_params_))
else: 
    parameters = {'bootstrap': False, 'min_samples_leaf': 3, 'n_estimators': 50, 
                  'min_samples_split': 3, 'max_features': 'log2', 'max_depth': 8}
    
    model = RandomForestClassifier(**parameters)
    model.fit(train_X, train_y)
    print (model.score( train_X , train_y ) , model.score( valid_X , valid_y ))

output = model.predict(test_X).astype(int)
df_output = pd.DataFrame()
aux = pd.read_csv(test_file)
df_output['PassengerId'] = aux['PassengerId']
df_output['Survived'] = output
df_output[['PassengerId', 'Survived']].to_csv(output_file, index=False)