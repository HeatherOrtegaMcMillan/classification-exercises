
import pandas as pd
import numpy as np
from env import host, password, user
import os

# Make a function named get_titanic_data that returns the titanic data from the codeup data science database as a pandas data frame. 
# Obtain your data from the Codeup Data Science Database.
############################## Functions for Aquiring Titanic Data ##############################

def get_db_url(db_name, user=user, host=host, password=password):
    """
        This helper function takes as default the user host and password from the env file.
        You must input the database name. It returns the appropriate URL to use in connecting to a database.
    """
    url = f'mysql+pymysql://{user}:{password}@{host}/{db_name}'
    return url

def get_db_data():
    '''
    This function reads data from the Codeup db into a df.
    '''
    sql_query =  "SELECT * FROM passengers"
    return pd.read_sql(sql_query, get_db_url('titanic_db'))

# UNCOMMENT TO TEST
# df = get_db_data()
# print(df.head())

# Make a function named get_iris_data that returns the data from the iris_db on the codeup data science database as a pandas data frame. 
# The returned data frame should include the actual name of the species in addition to the species_ids. 
# Obtain your data from the Codeup Data Science Database.
############################## Functions for Aquiring iris data ##############################
def get_db_url(db_name, user=user, host=host, password=password):
    """
        This helper function takes as default the user host and password from the env file.
        You must input the database name. It returns the appropriate URL to use in connecting to a database.
    """
    url = f'mysql+pymysql://{user}:{password}@{host}/{db_name}'
    return url

def get_db_data():
    '''
    This function reads data from the Codeup db into a df.
    '''
    sql_query =  "SELECT * FROM species JOIN measurements USING(species_id)"
    return pd.read_sql(sql_query, get_db_url('iris_db'))
    
# UNCOMMENT TO TEST
# df_iris = get_db_data() 
# print(df_iris.head())


# Once you've got your get_titanic_data and get_iris_data functions written, now it's time to add caching to them. 
# To do this, edit the beginning of the function to check for a local filename like titanic.csv or iris.csv. 
# If they exist, use the .csv file. 
# If the file doesn't exist, then produce the SQL and pandas necessary to create a dataframe, then write the dataframe to a .csv file with the appropriate name.
############################## Extra fancy check for csv stuff ##############################

# use os to check to see if df_titanic.csv exisits
def get_titanic_data():
    """
    This function loads the titanic data into a dataframe. If the file is cached as df_titanic.csv it will pull from the cached file.
    If not it will query the database and create the dataframe in that way. 
    Be sure the cashed file is located in the directory you are working in. 
    """
    if os.path.isfile('df_titanic.csv') == True:
        df = pd.read_csv('df_titanic.csv', index_col=0)
    else:
        sql_query =  "SELECT * FROM passengers"
        df = pd.read_sql(sql_query, get_db_url('titanic_db'))
    return df


def get_iris_data():
    """
    This function loads the iris data into a dataframe. If the file is cached as df_iris.csv it will pull from the cached file.
    If not it will query the database and create the dataframe in that way. 
    Be sure the cashed file is located in the directory you are working in. 
    """
    if os.path.isfile('df_iris.csv') == True:
        df = pd.read_csv('df_iris.csv', index_col=0)
    else:
        sql_query =  "SELECT * FROM species JOIN measurements USING(species_id)"
        df = pd.read_sql(sql_query, get_db_url('iris_db'))
    return df

