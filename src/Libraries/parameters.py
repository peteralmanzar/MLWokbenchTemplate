import sys
from typing import List, Union
from pandas import DataFrame
from sklearn.pipeline import Pipeline
from tensorflow.keras.models import Model

sys.path.append("./Libraries")
from mltoolkit import datatransformers, modeltemplates, persistence, modeltunning

dataFilePath: str = './Data/Raw/data.csv'
targetColumns: Union[str, List[str]] = 'target'
randomStateSeed: int = 42

pipelinePersistFitted: bool = False
trainingDataPersist: bool = False

splitStratisfy: bool = False
splitTestSize: float = 0.2

modelEpochs: int = 100
modelBatchSize: int = 32
modelEarlyStop: int = 10
modelOptimizer: Optimizer = Optimizer.ADAM
modelPersistTrained: bool = False

preProcessingPipeLine: Pipeline = Pipeline([
        # Add preprocessing steps here
        # see mltoolkit for available transformers
        # i.e.: ('encoder', datatransformers.transformer())    
    ])

model: Model = modeltemplates.getModelTemplateNone()

hyperParameters: dict = {
    # Add hyperparameters here
}