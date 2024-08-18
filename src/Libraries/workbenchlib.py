import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from typing import List, Tuple, Union
from pandas import DataFrame
from sklearn.pipeline import Pipeline
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model

sys.path.append("./Libraries")
import parameters
from mltoolkit import datatransformers, modeltunning, persistence

def csvToDataFrame() -> DataFrame:
    '''
    Reads csv file and returns DataFrame
    '''
    return pd.read_csv(parameters.dataFilePath)

def createPreprocessingPipeline() -> Pipeline:
    return parameters.getPreprocessingPipeline()

def fitPreprocessingPipeline(pipeline: Pipeline, data: DataFrame) -> None:
    '''
    Fits pipeline using raw data. If persist is True, saves pipeline to disk
    '''
    pipeline.fit(data)

    if parameters.pipelinePersistFitted:
        persistence.savePipeline(pipeline)

def processData(pipeline: Pipeline, data: DataFrame) -> DataFrame:
    '''
    Preprocesses data.
    '''
    return pipeline.transform(data)

def splitData(data: Union[DataFrame, List[Tuple[DataFrame, np.ndarray]]]) -> tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
    '''
    Splits data into training and testing sets
    '''
    if isinstance(data, DataFrame):
        return datatransformers.splitData(data, parameters.targetColumns, test_size=parameters.splitTestSize)
    else:
        X_train, X_test, y_train, y_test = datatransformers.splitSequencedData(data, parameters.targetColumns, test_size=parameters.splitTestSize)
        return X_train, X_test, y_train, y_test

def initModel() -> Model:
    '''
    Initializes model
    '''
    return parameters.getModel()

def tuneModel(model: Model, X_train: DataFrame, y_train: DataFrame, X_test: DataFrame, y_test: DataFrame) -> Model:
    '''
    Tunes model
    '''
    return modeltunning.tuneModel(model, parameters.hyperParameters, X_train, y_train, X_test, y_test)

def trainModel(model: Model, X_train: DataFrame, y_train: DataFrame, X_test: DataFrame, y_test: DataFrame) -> any:
    '''
    Trains model. If persist is True, saves model to disk
    '''
    earlyStop = EarlyStopping(monitor='val_loss', patience=parameters.modelEarlyStop, restore_best_weights=True, verbose=1)
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=parameters.modelEpochs, batch_size=parameters.modelBatchSize, callbacks=[earlyStop])

    if parameters.modelPersistTrained:
        persistence.saveModel(model)

    if parameters.trainingDataPersist:
        persistence.saveData(X_train, y_train, X_test, y_test)

    return history

def plotModelTrainingResults(history: any) -> None:
    '''
    Plots model training results
    '''
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def evaluateModel(model: Model, Xtest: DataFrame, yTest: DataFrame) -> tuple[float, float]:
    '''
    Evaluates model

    params:
    model: Model - trained model
    Xtest: DataFrame - test features
    yTest: DataFrame - test target
    '''
    return model.evaluate(Xtest, yTest)