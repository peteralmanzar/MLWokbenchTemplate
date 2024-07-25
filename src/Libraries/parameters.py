import sys
from pandas import DataFrame
from sklearn.pipeline import Pipeline
from tensorflow.keras.models import Model

sys.path.append("./Libraries")
from mltoolkit import datatransformers, modeltemplates, persistence, modeltunning

# Define the preprocessing pipeline
def getPreprocessingPipeline() -> Pipeline:
    '''
    Creates preprocessing pipeline
    '''
    pipeline = Pipeline([
        # Add preprocessing steps here
        # see mltoolkit for available transformers
        # i.e.: ('encoder', datatransformers.transformer())    
    ])

    return pipeline;

# Define the model
def getModel() -> Model:
    '''
    Initializes model
    '''
    # see mltoolkit for available model templates
    # or create your own model
    return Sequential([])

def getHyperparameters() -> dict:
    '''
    Returns hyperparameters
    '''
    return {
        # Add hyperparameters here
    }