# Disaster Response Pipeline

### Table of contents

## Introduction

This repository contains the information and code needed to create an app that classify a message related to a disaster into categories. 
The app uses a Machine Learning classifier trained with a large dataset of messages an categories, provided by [Figure Eight](https://www.figure-eight.com/).


Front page of the app:
![Screenshot 1](https://github.com/carboML/Disaster_Response_Pipeline/blob/main/cap_1.JPG)



## Installations

- pandas
- sys
- json
- plotly
- sqlite3
- requests
- sqlalchemy
- pickle
- Flask
- sklearn
- nltk

## File descriptions 
    
- `ETL Pipeline Preparation.ipynb`: Jupyter notebook to help ensemble the `process_data.py` file
- `ML Pipeline Preparation.ipynb`: Jupyter notebook to help ensemble the `train_classifier.py` file


- Data:

    - `disaster_messages.csv`: Dataset full a disaster messages
    - `disaster_categories.csv`: Dataset containing the categories of each disaster message 
    - `process_data.py`: The script takes the file paths of the two datasets and database, cleans the datasets, and stores the clean data into a SQLite database in the specified database file path.
 
- Model:
    - `train_classifier.py`:The script takes the database file path and model file path, creates and trains a classifier, and stores the classifier into a pickle file to the specified model file path.

- App: Foulder containing all the code for the app


## Instructions 

To execute the app follow the instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Author, and licensing
The app code and app andjupyter notebook templates, is given by Udacity as part as their Nanodegree in data science. The data set is provided by Figure 8
