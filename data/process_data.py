import sys
import pandas as pd
from sqlalchemy import create_engine
import numpy as np


def load_data(messages_filepath, categories_filepath):
    '''
    INPUTS: 
        messages_filepath:   filepath where the .csv containing the messages is
        categories_filepath: filepath where the .csv containing the messages categories is
    OUTPUT:
        df: Pandas dataframe with both .csv merged
    '''



    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on="id")
    

    return df



def clean_data(df):
    
    '''
    INPUT:
        df: Pandas dataframe with both .csv merged
    OUTPUT:
        df: The same pandas dataframe clean
    '''

    categories = df['categories'].str.split(pat=';', n=-1, expand=True) 

    row = categories.iloc[0,:] # The first row of the categories dataframe

    category_colnames = [x[:-2] for x in row] # This "for" takes away the last two characters of each element in the list

    categories.columns = category_colnames

    for column in categories:
    
        # set each value to be the last character of the string
    
        categories[column] = categories[column].astype(str).str[-1] # str[-1] takes the first character starting from the rigth
        # convert column from string to numeric
        categories[column] = categories[column].astype(np.int)
    
    # Join the clean categories column with df
    df['related'].replace(1,2,inplace=True)
    df.drop(columns = ['categories'],  inplace=True)

    df = pd.concat([df, categories],axis=1)
    # Drop the duplicates values
    df.drop_duplicates(keep=False,inplace =True )

    return df


def save_data(df, database_filename):
    
    '''
    This function save the pandas dataframe into an SQL database
    
    INPUTS:
        df: pandas dataframe
        database_filename: Filename name of the database
    '''
        
    

    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('table_messages', engine, index=False,if_exists='replace')  



def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()