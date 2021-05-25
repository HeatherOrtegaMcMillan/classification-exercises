import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import classification_report, confusion_matrix

######################################## Iris ########################################################

def iris_split(df):
    '''
    This function take in the iris data acquired by aquire.py, get_iris_data,
    performs a split, stratifies by species.
    Returns train, validate, and test dfs.
    '''
    train_validate, test = train_test_split(df, test_size=.2, 
                                        random_state=713,
                                        stratify = df.species)
    train, validate = train_test_split(train_validate, test_size=.3, 
                                   random_state=713,
                                   stratify= train.species)
    return train, validate, test


def prep_iris(df):
    """
    This function takes the iris data set (from aquire.py) as input.
    And outputs a clean dataset ready for modeling.
    Clean in this case means: species and measurement id columns removed, species_name changed to species and seperated 
    into dummy variables
    """
    cleaned_df = df.drop(columns = ['species_id', 'measurement_id'])
    cleaned_df = cleaned_df.rename(columns= {'species_name': 'species'})
    cleaned_df = pd.get_dummies(data=cleaned_df, columns= ['species'], drop_first = True)
    return cleaned_df

def prep_iris_explore(df):
    """
    This function takes the iris data set (from aquire.py) as input.
    And outputs a clean dataset, split into train, validate and test sections ready for exploring.
    Clean in this case means: species and measurement id columns removed, species_name remains (as species) 
    for stratifying during splitting.
    """
    cleaned_df = df.drop(columns = ['species_id', 'measurement_id'])
    cleaned_df = cleaned_df.rename(columns= {'species_name': 'species'})
    return cleaned_df

def train_validate_test_split(df, target, seed=713):
    '''
    This function takes in a dataframe, the name of the target variable
    (for stratification purposes), and an integer for a setting a seed
    and splits the data into train, validate and test. 
    Test is 20% of the original dataset, validate is .30*.80= 24% of the 
    original dataset, and train is .70*.80= 56% of the original dataset. 
    The function returns, in this order, train, validate and test dataframes. 
    '''
    train_validate, test = train_test_split(df, test_size=0.2, 
                                            random_state=seed, 
                                            stratify=df[target])
    train, validate = train_test_split(train_validate, test_size=0.3, 
                                       random_state=seed,
                                       stratify=train_validate[target])
    return train, validate, test

######################################## Titanic ########################################################

def titanic_split(df):
    '''
    This function take in the titanic data acquired by get_titanic_data,
    performs a split and stratifies survived column.
    Returns train, validate, and test dfs.
    '''
    train_validate, test = train_test_split(df, test_size=.2, 
                                        random_state=123, 
                                        stratify=df.survived)
    train, validate = train_test_split(train_validate, test_size=.3, 
                                   random_state=123, 
                                   stratify=train_validate.survived)
    return train, validate, test



def impute_mean_age(train, validate, test):
    '''
    This function imputes the mean of the age column for
    observations with missing values.
    Returns transformed train, validate, and test df.
    '''
    # create the imputer object with mean strategy
    imputer = SimpleImputer(strategy = 'mean')
    
    # fit on and transform age column in train
    train['age'] = imputer.fit_transform(train[['age']])
    
    # transform age column in validate
    validate['age'] = imputer.transform(validate[['age']])
    
    # transform age column in test
    test['age'] = imputer.transform(test[['age']])
    
    return train, validate, test


def prep_titanic(df):
    '''
    This function take in the titanic data acquired by get_titanic_data,
    Returns prepped train, validate, and test dfs with embarked dummy vars,
    deck dropped, and the mean of age imputed for Null values.
    '''
    
    # drop rows where embarked/embark town are null values
    df = df[~df.embarked.isnull()]
    
    # encode embarked using dummy columns
    titanic_dummies = pd.get_dummies(df.embarked, drop_first=True)
    
    # join dummy columns back to df
    df = pd.concat([df, titanic_dummies], axis=1)
    
    # drop the deck column
    df = df.drop(columns='deck')
    
    # split data into train, validate, test dfs
    train, validate, test = titanic_split(df)
    
    # impute mean of age into null values in age column
    train, validate, test = impute_mean_age(train, validate, test)
    
    return train, validate, test

def impute_mode(train, validate, test):
    '''
    impute mode for embark_town
    '''
    imputer = SimpleImputer(strategy='most_frequent', missing_values=None)
    train['embark_town'] = imputer.fit_transform(train[['embark_town']])
    validate['embark_town'] = imputer.transform(validate[['embark_town']])
    test['embark_town'] = imputer.transform(test[['embark_town']])
    return train, validate, test

def prep_titanic_data(df):
    '''
    takes in a dataframe of the titanic dataset as it is acquired and returns a cleaned dataframe
    arguments: df: a pandas DataFrame with the expected feature names and columns
    return: train, test, split: three dataframes with the cleaning operations performed on them
    '''
    df = df.drop_duplicates()
    df = df.drop(columns=['deck', 'embarked', 'class', 'age', 'passenger_id'])
    train, test = train_test_split(df, test_size=0.2, random_state=1349, stratify=df.survived)
    train, validate = train_test_split(train, train_size=0.7, random_state=1349, stratify=train.survived)
    train, validate, test = impute_mode(train, validate, test)
    train = pd.get_dummies(data = train, columns=['sex', 'embark_town'], drop_first=[True,True])
    validate = pd.get_dummies(data=validate, columns=['sex', 'embark_town'], drop_first=[True,True])
    test = pd.get_dummies(data=test, columns = ['sex', 'embark_town'], drop_first=[True,True])
    return train, validate, test



###########################################################################################


def all_aboard_the_X_train(X_cols, y_col, train, validate, test):
    '''
    X_cols = list of column names you want as your features
    y_col = string that is the name of your target column
    train = the name of your train dataframe
    validate = the name of your validate dataframe
    test = the name of your test dataframe
    outputs X_train and y_train, X_validate and y_validate, and X_test and y_test
    6 variables come out! So have that ready
    '''
    
    # do the capital X lowercase y thing for train test and split
    # X is the data frame of the features, y is a series of the target
    X_train, y_train = train[X_cols], train[y_col]
    X_validate, y_validate = validate[X_cols], validate[y_col]
    X_test, y_test = test[X_cols], test[y_col]
    
    return X_train, y_train, X_validate, y_validate, X_test, y_test

    
    