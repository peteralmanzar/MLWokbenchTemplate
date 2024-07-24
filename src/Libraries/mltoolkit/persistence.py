import joblib
import pandas as pd
from datetime import datetime
from keras.models import Model
from pandas import DataFrame
from sklearn.pipeline import Pipeline
from typing import Any, Union

def savePipeline(pipeline: Pipeline) -> str:
    '''
    Save pipeline to file

    Parameters:
    pipeline (Pipeline): Pipeline to save

    Returns:
    fileName (str): Name of the saved file
    '''
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    fileName = f"./Persistence/Pipelines/pipeline_{current_time}.pkl"
    joblib.dump(pipeline, fileName)
    return fileName

def saveModel(model: Model) -> None:
    '''
    Save model to file

    Parameters:
    model (Model): Model to save

    Returns:
    fileName (str): Name of the saved file
    '''
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    fileName = f"./Persistence/Models/model_{current_time}.pkl"
    joblib.dump(model, fileName)
    return fileName

def saveData(Xtrain: DataFrame, yTrain: DataFrame, X_test: DataFrame, y_test: DataFrame) -> None:
    '''
    Save training data to file

    Parameters:
    Xtrain (DataFrame): Training data
    yTrain (DataFrame): Training target

    Returns:
    Tuple[str, str]: File names of the saved files
    '''
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    xTrainFileName = f"./Data/Training/training_data_features_{current_time}.csv"
    yTrainFileName = f"./Data/Training/training_data_target_{current_time}.csv"
    Xtrain.to_csv(xTrainFileName, index=False)
    yTrain.to_csv(yTrainFileName, index=False)
    return xTrainFileName, yTrainFileName

def LoadPipeline(fileName: str) -> Pipeline:
    '''
    Load pipeline from file

    Parameters:
    fileName (str): Name of the file(extension included)
    '''
    return joblib.load(fileName)

def LoadModel(fileName: str) -> Model:
    '''
    Load ML model from file

    Parameters:
    fileName (str): Name of the file(extension included)
    '''
    return joblib.load(fileName)

def LoadData(XTrainFileName: str, yTrainFileName: str, XTestFileName: str, yTestFileName) -> tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
    '''
    Load training data from file

    Parameters:
    XtrainFileName (str): Name of the training data file
    yTrainFileName (str): Name of the training target file
    XTestFileName (str): Name of the test data file
    yTestFileName (str): Name of the test target file
    '''
    Xtrain = pd.read_csv(XTrainFileName)
    yTrain = pd.read_csv(yTrainFileName)
    XTest = pd.read_csv(XTestFileName)
    yTest = pd.read_csv(yTestFileName)
    return Xtrain, yTrain, XTest, yTest