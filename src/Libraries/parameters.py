import sys
from typing import List, Union
from pandas import DataFrame
from sklearn.pipeline import Pipeline
from tensorflow.keras.models import Model

sys.path.append("./Libraries")
from mltoolkit import datatransformers, modeltemplates, persistence

class MlWorkbenchParameters:
    def __init__(self) -> None:
        self.dataFilePath: str = "./"
        self.targetColumns: Union[str, List[str]] = ""
        self.randomStateSeed: int = 42

        self.pipelinePersistFitted: bool = False    
        self.trainingDataPersist: bool = False

        self.splitStratisfy: bool = False
        self.splitTestSize: float = 0.2

        self.modelEpochs: int = 100
        self.modelBatchSize: int = 32
        self.modelEarlyStop: int = 10
        self.modelOptimizer: modeltemplates.Optimizer = modeltemplates.Optimizer.ADAM
        self.modelPersistTrained: bool = False

        self.preProcessingPipeLine: Pipeline = Pipeline([
            # Add preprocessing steps here
            # see mltoolkit for available transformers
            # i.e.: ('encoder', datatransformers.transformer())    
        ])

        self.model: Model = modeltemplates.getModelTemplateNone()

        self.hyperParameters: dict = None