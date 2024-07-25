import pandas as pd
import matplotlib.pyplot as plt
import sys
from pandas import DataFrame
from sklearn.pipeline import Pipeline
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model

sys.path.append("./Libraries")
from mltoolkit import datatransformers, modeltemplates, modeltunning, parameters, persistence

def createPreprocessingPipeline() -> Pipeline:
    return parameters.getPreprocessingPipeline()

def initModel() -> Model:
    '''
    Initializes model
    '''
    return parameters.getModel()

def tuneModel(model: Model, X_train: DataFrame, y_train: DataFrame, X_test: DataFrame, y_test: DataFrame, ) -> Model:
    '''
    Tunes model
    '''
    hyperparameters = parameters.getHyperparameters()
    return modeltunning.tuneModel(model, hyperparameters, X_train, y_train, X_test, y_test)

def csvToDataFrame(path: str) -> DataFrame:
    '''
    Reads csv file and returns DataFrame
    '''
    return pd.read_csv(path)

def fitPreprocessingPipeline(pipeline: Pipeline, data: DataFrame, persist: bool = False) -> None:
    '''
    Fits pipeline using raw data. If persist is True, saves pipeline to disk
    '''
    pipeline.fit(data)

    if persist:
        persistence.savePipeline(pipeline)

def processData(pipeline: Pipeline, data: DataFrame) -> DataFrame:
    '''
    Preprocesses data.
    '''
    return pipeline.transform(data)

def splitData(data: DataFrame, target: str, test_size: float = 0.2) -> tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
    '''
    Splits data into training and testing sets
    '''
    return datatransformers.splitData(data, target)

def trainModel(model: Model, X_train: DataFrame, y_train:DataFrame, X_test: DataFrame, y_test: DataFrame, epochs: int=100, batchSize: int= 32, earlyStop: int= 10, persistModel: bool = False, persistTrainingData: bool = False) -> any:
    '''
    Trains model. If persist is True, saves model to disk
    '''
    earlyStop = EarlyStopping(monitor='val_loss', patience=earlyStop, restore_best_weights=True, verbose=1)
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batchSize, callbacks=[earlyStop])

    if persistModel:
        persistence.saveModel(model)

    if persistTrainingData:
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

def evaluateModel(model: any, Xtest: DataFrame, yTest: DataFrame) -> tuple[float, float]:
    '''
    Evaluates model

    params:
    model: Model - trained model
    Xtest: DataFrame - test features
    yTest: DataFrame - test target
    '''
    return model.evaluate(Xtest, yTest)