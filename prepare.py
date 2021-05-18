import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

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