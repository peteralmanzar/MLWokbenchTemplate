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

class MlWorkbench:
    def __init__(self, parameters: parameters.MlWorkbenchParameters) -> None:
        self.parameters = parameters

    def csvToDataFrame(self) -> DataFrame:
        '''
        Reads csv file and returns DataFrame
        '''
        return pd.read_csv(self.parameters.dataFilePath)
    
    def createPreprocessingPipeline(self) -> Pipeline:
        return self.parameters.getPreprocessingPipeline()
    
    def fitPreprocessingPipeline(self, pipeline: Pipeline, data: DataFrame) -> None:
        '''
        Fits pipeline using raw data. If persist is True, saves pipeline to disk
        '''
        pipeline.fit(data)

        if self.parameters.pipelinePersistFitted:
            persistence.savePipeline(pipeline)

    def processData(self, pipeline: Pipeline, data: DataFrame) -> DataFrame:
        '''
        Preprocesses data.
        '''
        return pipeline.transform(data)
    
    def splitData(self, data: Union[DataFrame, List[Tuple[DataFrame, np.ndarray]]]) -> tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
        '''
        Splits data into training and testing sets
        '''
        if isinstance(data, DataFrame):
            return datatransformers.splitData(self, data, test_size=self.parameters.splitTestSize)
        else:
            X_train, X_test, y_train, y_test = datatransformers.splitSequencedData(data, self.parameters.targetColumns, test_size=self.parameters.splitTestSize)
            return X_train, X_test, y_train, y_test
        
    def initModel(self) -> Model:
        '''
        Initializes model
        '''
        return self.parameters.getModel()
    
    def tuneModel(self, model: Model, X_train: DataFrame, y_train: DataFrame, X_test: DataFrame, y_test: DataFrame) -> Model:
        '''
        Tunes model
        '''
        return modeltunning.tuneModel(model, self.parameters.hyperParameters, X_train, y_train, X_test, y_test)
    
    def trainModel(self, model: Model, X_train: DataFrame, y_train: DataFrame, X_test: DataFrame, y_test: DataFrame) -> any:
        '''
        Trains model. If persist is True, saves model to disk
        '''
        earlyStop = EarlyStopping(monitor='val_loss', patience=self.parameters.modelEarlyStop, restore_best_weights=True, verbose=1)
        history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=self.parameters.modelEpochs, batch_size=self.parameters.modelBatchSize, callbacks=[earlyStop])

        if self.parameters.modelPersistTrained:
            persistence.saveModel(model)

        if self.parameters.trainingDataPersist:
            persistence.saveData(X_train, y_train, X_test, y_test)

        return history
    
    def plotModelTrainingResults(self, history: any) -> None:
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

    def evaluateModel(self, model: Model, Xtest: DataFrame, yTest: DataFrame) -> tuple[float, float]:
        '''
        Evaluates model

        params:
        model: Model - trained model
        Xtest: DataFrame - test features
        yTest: DataFrame - test target
        '''
        return model.evaluate(Xtest, yTest)
    
    def predict(self, model: Model, data: DataFrame) -> np.ndarray:
        '''
        Predicts target using model
        '''
        return model.predict(data)
    
    def run(self) -> None:
        '''
        Runs the workbench
        '''
        data = self.csvToDataFrame()
        pipeline = self.createPreprocessingPipeline()
        self.fitPreprocessingPipeline(pipeline, data)
        processedData = self.processData(pipeline, data)
        X_train, X_test, y_train, y_test = self.splitData(processedData)
        model = self.initModel()
        model = self.tuneModel(model, X_train, y_train, X_test, y_test)
        history = self.trainModel(model, X_train, y_train, X_test, y_test)
        self.plotModelTrainingResults(history)
        loss, accuracy = self.evaluateModel(model, X_test, y_test)
        print(f"Loss: {loss}, Accuracy: {accuracy}")
        print(y_test)
        print("Done!")