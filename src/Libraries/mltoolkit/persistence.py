from typing import Any
import joblib

def Save(fileName: str, modelOrPipeline: Any) -> None:
    '''
    Save ML model or pipeline to file
    '''
    # Save the model to disk
    joblib.dump(modelOrPipeline, fileName + '.pkl')

def Load(fileName: str) -> Any:
    '''
    Load ML model or pipeline from file
    '''
    # Load the model from disk
    return joblib.load(fileName + '.pkl')