import sys
import pandas as pd
from pandas import DataFrame
from sklearn.pipeline import Pipeline

sys.path.append("./Libraries")

def csvToDataFrame(path: str) -> DataFrame:
    '''
    Reads csv file and returns DataFrame
    '''
    return pd.read_csv(path)

def createPreprocessingPipeline() -> Pipeline:
    '''
    Creates preprocessing pipeline
    '''
    return Pipeline()

def fitPreprocessingPipeline(pipeline: Pipeline, data: DataFrame, persist: bool = False):
    '''
    Fits pipeline using raw data. If persist is True, saves pipeline to disk
    '''
    pass 

def preProcessData(pipeline: Pipeline, data: DataFrame) -> DataFrame:
    '''
    Preprocesses data. If persist is True
    '''
    pass

def splitData(data: DataFrame, target: str, test_size: float = 0.2) -> tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
    '''
    Splits data into training and testing sets
    '''
    pass

def initModel() -> any:
    '''
    Initializes model
    '''
    pass

def trainModel(model: any, persistModel: bool = False, persistTrainingData: bool = False) -> None:
    '''
    Trains model. If persist is True, saves model to disk
    '''
    pass

def plotModelTrainingResults(model: any) -> None:
    '''
    Plots model training results
    '''
    pass

def evaluateModel(model: any, data: DataFrame) -> tuple[float, float]:
    '''
    Evaluates model
    '''
    pass