import pandas as pd
import numpy as np

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