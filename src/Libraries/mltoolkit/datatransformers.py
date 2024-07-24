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
        A new DataFrame with the encoded columns
    """

    def __init__(self, columns: List[str]):
        self.columns = columns
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
        self.strategy = strategy
        self.columns = columns

    def fit(self, X: DataFrame, y: Union[DataFrame, None] = None) -> 'PipelineImputer':
        return self
    
    def transform(self, X: DataFrame) -> DataFrame:
        for column in self.columns:
            if column not in X.columns:
                raise ValueError(f"Column '{column}' not found in DataFrame")
                
            if self.strategy == ImputeStrategy.MEAN:
                if X[column].dtype.kind in 'biufc':  # Check if the column is of numeric type
                    print(f"Imputing column '{column}' with mean value")
                    X[column] = X[column].fillna(X[column].mean())
                else:
                    raise ValueError(f"Mean imputation is not suitable for column '{column}' with dtype {X[column].dtype}")
            elif self.strategy == ImputeStrategy.MEDIAN:
                if X[column].dtype.kind in 'biufc':  # Check if the column is of numeric type
                    print(f"Imputing column '{column}' with median value")
                    X[column] = X[column].fillna(X[column].median())
                else:
                    raise ValueError(f"Median imputation is not suitable for column '{column}' with dtype {X[column].dtype}")
            elif self.strategy == ImputeStrategy.MODE:
                mode_value = X[column].mode()
                if not mode_value.empty:
                    print(f"Imputing column '{column}' with mode value")
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
        self.columns = columns

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
        self.columns = columns
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
        self.columns = columns
        self.dropColumns = dropColumns

    def fit(self, X: DataFrame, y: Union[DataFrame, None] = None) -> 'PipelineDateSpliter':
        return self
    
    def transform(self, X: DataFrame) -> DataFrame:
        for column in self.columns:
            X[column] = pd.to_datetime(X[column], errors='coerce')
            X[column + "_year"] = X[column].dt.year
            X[column + "_month"] = X[column].dt.month
            X[column + "_day"] = X[column].dt.day
            if self.dropColumns:
                X = X.drop([column], axis=1)
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
        self.index = index

    def fit(self, X: DataFrame, y: Union[DataFrame, None] = None) -> 'PipelineIndexSetter':
        return self
    
    def transform(self, X: DataFrame) -> DataFrame:
        return X.set_index(self.index)

class PipelineSequencer(BaseEstimator, TransformerMixin):
    """
    PipelineSequencer is a custom transformer that creates sequences from a DataFrame.
    It takes a column to use as the label and a sequence length as parameters.

    Parameters
    ----------
    column : str
        A column name to use as the label.
    sequence_length : int
        The length of the sequence to create.

    Returns
    -------
    tuple
        A tuple containing a list of DataFrames and a DataFrame.
    """

    def __init__(self, column: str, sequence_length: int = 60):
        self.column = column
        self.sequence_length = sequence_length

    def fit(self, X: DataFrame, y: Union[DataFrame, None] = None) -> 'PipelineSequencer':
        return self
    
    def transform(self, X: DataFrame) -> Tuple[List[DataFrame], DataFrame]:
        return self.create_sequences(X)
        
    def create_sequences(self, data: DataFrame) -> Tuple[List[DataFrame], DataFrame]:
        sequencedX = []
        sequencedy = []
        for i in range(self.sequence_length, len(data)):
            sequence_df = data.iloc[i-self.sequence_length:i].copy()
            sequencedX.append(sequence_df)
            sequencedy.append(data.iloc[i][self.column])
                        
        label_df = pd.DataFrame(sequencedy, columns=['label'])
        return sequencedX, label_df
    
def splitData(data: DataFrame, target: str, test_size: float = 0.2, seed: int = 42) -> Tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
    X = data.drop(target, axis=1)
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    return X_train, X_test, y_train, y_test

def dataSplitToNumpyArray(X_train: DataFrame, X_test: DataFrame, y_train: DataFrame, y_test: DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return X_train.to_numpy(), X_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()