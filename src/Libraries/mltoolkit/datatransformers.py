from enum import Enum
import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder, StandardScaler
from typing import List, Tuple, Union

class ImputeStrategy(Enum):
    MEAN = 'mean'
    MEDIAN = 'median'
    MODE = 'mode'

class PipelineOneHotEncoder(BaseEstimator, TransformerMixin):
    """
    PipelineOneHotEncoder is a custom transformer that encodes categorical columns using one-hot encoding.
    It is a wrapper around the OneHotEncoder from the sklearn library.
    It takes a list of columns to encode as a parameter.

    Parameters
    ----------
    columns : list
        A list of column names to encode.

    Returns
    -------
    DataFrame
        A new DataFrame with the specified columns encoded.
    """

    def __init__(self, columns: List[str]):
        self.columns: List[str] = columns
        self.encoder = SklearnOneHotEncoder(sparse=False, drop='first')

    def fit(self, X: DataFrame, y: Union[DataFrame, None] = None) -> 'PipelineOneHotEncoder':
        self.encoder.fit(X[self.columns])
        return self
    
    def transform(self, X: DataFrame) -> DataFrame:
        encoded = self.encoder.transform(X[self.columns])
        encoded_df = DataFrame(encoded, columns=self.encoder.get_feature_names_out(self.columns))
        X = X.drop(self.columns, axis=1)
        X = X.reset_index(drop=True)
        encoded_df = encoded_df.reset_index(drop=True)
        return X.join(encoded_df)

class PipelineImputer(BaseEstimator, TransformerMixin):
    """
    PipelineImputer is a custom transformer that imputes missing values in columns using a specified strategy.
    It takes a strategy and a list of columns to impute as parameters.

    Parameters
    ----------
    strategy : ImputeStrategy
        The imputation strategy to use.
    columns : list
        A list of column names to impute.

    Returns
    -------
    DataFrame
        A new DataFrame with the specified columns imputed.
    """
    def __init__(self, strategy: ImputeStrategy, columns: List[str]):
        self.strategy: ImputeStrategy = strategy
        self.columns: List[str] = columns

    def fit(self, X: DataFrame, y: Union[DataFrame, None] = None) -> 'PipelineImputer':
        return self
    
    def transform(self, X: DataFrame) -> DataFrame:
        for column in self.columns:
            if column not in X.columns:
                raise ValueError(f"Column '{column}' not found in DataFrame")
                
            if self.strategy == ImputeStrategy.MEAN:
                if X[column].dtype.kind in 'biufc':  # Check if the column is of numeric type
                    X[column] = X[column].fillna(X[column].mean())
                else:
                    raise ValueError(f"Mean imputation is not suitable for column '{column}' with dtype {X[column].dtype}")
            elif self.strategy == ImputeStrategy.MEDIAN:
                if X[column].dtype.kind in 'biufc':  # Check if the column is of numeric type
                    X[column] = X[column].fillna(X[column].median())
                else:
                    raise ValueError(f"Median imputation is not suitable for column '{column}' with dtype {X[column].dtype}")
            elif self.strategy == ImputeStrategy.MODE:
                mode_value = X[column].mode()
                if not mode_value.empty:
                    X[column] = X[column].fillna(mode_value[0])
                else:
                    raise ValueError(f"No mode value found for column '{column}'")
            else:
                raise ValueError(f"Unsupported impute strategy: {self.strategy}")
        return X

class PipelineFeatureDropper(BaseEstimator, TransformerMixin):
    """
    PipelineFeatureDropper is a custom transformer that drops columns from a DataFrame.
    It takes a list of columns to drop as a parameter.

    Parameters
    ----------
    columns : list
        A list of column names to drop.

    Returns
    -------
    DataFrame
        A new DataFrame with the specified columns dropped.
    """

    def __init__(self, columns: List[str]):
        '''
        columns : list
            A list of column names to drop.
        '''
        self.columns: List[str] = columns

    def fit(self, X: DataFrame, y: Union[DataFrame, None] = None) -> 'PipelineFeatureDropper':
        return self
    
    def transform(self, X: DataFrame) -> DataFrame:
        return X.drop(self.columns, axis=1, errors="ignore")

class PipelineFeatureStandardScaler(BaseEstimator, TransformerMixin):
    """
    PipelineFeatureStandardScaler is a custom transformer that scales numerical columns using the StandardScaler from the sklearn library.
    It takes a list of columns to scale as a parameter.

    Parameters
    ----------
    columns : list
        A list of column names to scale.

    Returns
    -------
    DataFrame
        A new DataFrame with the specified columns scaled.
    """

    def __init__(self, columns: List[str]):
        self.columns: List[str] = columns
        self.scaler = StandardScaler()

    def fit(self, X: DataFrame, y: Union[DataFrame, None] = None) -> 'PipelineFeatureStandardScaler':
        self.scaler.fit(X[self.columns])
        return self
    
    def transform(self, X: DataFrame) -> DataFrame:
        X[self.columns] = self.scaler.transform(X[self.columns])
        return X

class PipelineDateSpliter(BaseEstimator, TransformerMixin):
    """
    PipelineDateSpliter is a custom transformer that splits date columns into day, month, and year columns.
    It takes a list of columns to split as a parameter.

    Parameters
    ----------
    columns : list
        A list of column names to split.

    Returns
    -------
    DataFrame
        A new DataFrame with the specified columns split.
    """

    def __init__(self, columns: List[str], dropColumns: bool = True):
        self.columns: List[str] = columns
        self.dropColumns: bool = dropColumns

    def fit(self, X: DataFrame, y: Union[DataFrame, None] = None) -> 'PipelineDateSpliter':
        return self
    
    def transform(self, X: DataFrame) -> DataFrame:
        for column in self.columns:
            X[column] = pd.to_datetime(X[column], errors='coerce')
            X[column + "_year"] = X[column].dt.year
            X[column + "_month"] = X[column].dt.month
            X[column + "_day"] = X[column].dt.day
            if self.dropColumns:
                X = X.drop(column, axis=1, errors='ignore')
        return X

class PipelineIndexSetter(BaseEstimator, TransformerMixin):
    """
    PipelineIndexSetter is a custom transformer that sets the index of a DataFrame.
    It takes a column name to use as the index as a parameter.

    Parameters
    ----------
    index : str
        A column name to use as the index.

    Returns
    -------
    DataFrame
        A new DataFrame with the specified column as the index.
    """

    def __init__(self, index: str):
        self.index: str = index

    def fit(self, X: DataFrame, y: Union[DataFrame, None] = None) -> 'PipelineIndexSetter':
        return self
    
    def transform(self, X: DataFrame) -> DataFrame:
        return X.set_index(self.index)

class PipelineSequencer(BaseEstimator, TransformerMixin):
    """
    PipelineSequencer is a custom transformer that creates sequences from a DataFrame.
    It takes a list of column names to use as the target values and a sequence length as parameters.

    Parameters
    ----------
    targetColumns : list
        A list of column names to use as the target values.
    sequence_length : int
        The length of the sequences to create.

    Returns
    -------
    tuple
        A tuple containing the features and target values.
    """

    def __init__(self, targetColumns: List[str], sequence_length: int = 60):
        self.targetColumns: List[str] = targetColumns
        self.sequence_length: int = sequence_length

    def fit(self, X: DataFrame, y: Union[DataFrame, None] = None) -> 'PipelineSequencer':
        return self
    
    def transform(self, X: DataFrame) -> List[Tuple[DataFrame, np.ndarray]]:
        return self.create_sequences(X) 
        
    def create_sequences(self, data: DataFrame) -> List[Tuple[DataFrame, np.ndarray]]:
        sequencedX = []
        sequencedy = []
        for i in range(self.sequence_length, len(data)):
            sequence_df = data.iloc[i-self.sequence_length:i].copy()
            sequencedX.append(sequence_df)
            sequencedy.append(data.iloc[i][self.targetColumns])

        target_df = pd.DataFrame(sequencedy, columns=self.targetColumns)
        X = [(features, target) for features, target in zip(sequencedX, target_df.values)]
        
        return X

def splitData(data: DataFrame, target_columns: List[str], test_size: float = 0.2, seed: int = 42) -> Tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
    X: DataFrame = data.drop(target_columns, axis=1)
    y: DataFrame = data[target_columns]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y, shuffle=False)
    return X_train, X_test, y_train, y_test 

def splitSequencedData(data: List[Tuple[DataFrame, np.ndarray]], target_columns: List[str], test_size: float = 0.2, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train, test = train_test_split(data, test_size=test_size, random_state=seed, shuffle=False)

    X_train = [item[0] for item in train]
    y_train = [item[1] for item in train]
    X_test = [item[0] for item in test]
    y_test = [item[1] for item in test]
    
    X_train = [sequence.drop(columns=target_columns, errors='ignore') for sequence in X_train]
    X_test = [sequence.drop(columns=target_columns, errors='ignore') for sequence in X_test]        

    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)