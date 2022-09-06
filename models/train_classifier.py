import sys
import pandas as pd
import pickle

from sqlalchemy import create_engine
import numpy as np
import nltk
nltk.download('stopwords')
import re
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from nltk.tokenize import word_tokenize 
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from sklearn.model_selection import GridSearchCV


def load_data(database_filepath):
    
    '''
    load data from database
    It creates two pandas, X ( inputs of the model) Y ( outputs of the model)
    
    
    INPUTS:
        database_filepath: Path where the database created in process_data.py is stored
    
    '''
    
    engine = create_engine('sqlite:///data/DisasterResponse.db')
    #Read the sql and store it in a pandas dataframe
    df = pd.read_sql_table('table_messages',engine)
    #Split the dataframe into inputs and ouputs
    X = df[df.columns[1]].astype(str)
    Y = df[df.columns[5:]].astype('int')

    return X,Y


def tokenize(text):
    '''
    This function tokenize text, it uses various steps
    
    INPUT:
        text: An string array
    OUTPUT:
        lemmed: Tokenize text
    
    '''
    
    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # Tokenize text
    words = text.split()
    
    # Remove stop words
    words = [w for w in words if w not in stopwords.words("english")]
    
    
    # Reduce words to their root form
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in words]

    return lemmed


def clasification_metrics(y_test,y_pred):
    
    ''' 
    Function created to simplify the evaluation step
    It will give the f1 score, precision and recall for each feature or category
    
    INPUTS:
        y_test: true output for a certain text
        y_pred: Predicted output for the same text
        
    OUTPUT:
        accuracy_list: An array with the accurary score for each feature
        
        
    '''
    i = 0  
    accuracy_list = []
    for col in y_test:
        print('Feature:',col)
        print(classification_report(y_test[col], y_pred[:, i]))
        accuracy = accuracy_score(y_test[col], y_pred[:, i], normalize=True)
        accuracy_list.append(accuracy)
      
    
    return accuracy_list

def build_model():
    
    # This function build a random forest classification model throw a ML pipeline

    pipe= Pipeline([
        ('vect', CountVectorizer(analyzer = 'word', tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf',  MultiOutputClassifier(RandomForestClassifier(n_estimators=5)))
    ])

    # Since the grid search takes forever to run in my personal computer, I'd rather create a model with 5 estimators that is not the optimal solution
    # If I had to run the cv grid everytime this project will take forever to be completed
   
    parameters = pipe.get_params()

    # After taking a look at this 
    parameters = {
            'tfidf__use_idf': (True, False), #rather to use or not tdif 
            'clf__estimator__n_estimators': [5, 6] # Number of estimator ( trees) from 5 to 6, Future users can increse this        range
    }

    cv = GridSearchCV(pipe, param_grid = parameters)
    

    return cv




    


def evaluate_model(model, X_test, Y_test):
    
    '''
    INPUTS:
        model: The model fitted 
        X_test: Inputs of the test dataset
        Y_test: Outputs of the test dataset
    
    OUTPUT:
        accuracy_listing: An array with the accurary score for each feature
    
    '''
    
    y_pred = model.predict(X_test)
    accuary_listing = clasification_metrics(Y_test,y_pred)
    print(np.mean(accuary_listing))

    return accuary_listing
    
    

def save_model(model, model_filepath):
    
    # This function saves the model on a pickle file
    
    with open (model_filepath, 'wb') as f:
        pickle.dump(model, f)



def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        print('SALE!')
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()