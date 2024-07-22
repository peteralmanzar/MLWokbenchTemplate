import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder, StandardScaler
from typing import List, Tuple, Union

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

class PipelineDataSplit(BaseEstimator, TransformerMixin):
    """
    DataSplit is a custom transformer that splits a DataFrame into training and testing sets.

    Parameters
    ----------
    labelColumns : list
        A list of column names to use as labels.
    testSize : float
        The proportion of the dataset to include in the test split.
    randomState : int
        The seed used by the random number generator.

    Returns
    -------
    tuple
        A tuple containing the training and testing sets.
    """

    def __init__(self, labelColumns: List[str], testSize: float = 0.2, randomState: int = 42):
        self.labelColumns = labelColumns
        self.testSize = testSize
        self.randomState = randomState

    def fit(self, X: DataFrame, y: Union[DataFrame, None] = None) -> 'DataSplit':
        return self
    
    def transform(self, X: DataFrame) -> Tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
        y = X[self.labelColumns]
        X = X.drop(self.labelColumns, axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.testSize, random_state=self.randomState)
        return X_train, X_test, y_train, y_test

class DatasplitToNumpyArray(BaseEstimator, TransformerMixin):
    """
    DatasplitToNumpyArray is a custom transformer that converts the data split into numpy arrays.
    It takes a tuple of DataFrames as input and returns a tuple of numpy arrays.

    Parameters
    ----------
    X : tuple
        A tuple of DataFrames containing the training and testing data.
    y : DataFrame
        A DataFrame containing the target data.

    Returns
    -------
    tuple
        A tuple of numpy arrays containing the training and testing data.
    """

    def fit(self, X: Tuple[DataFrame, DataFrame, DataFrame, DataFrame], y: Union[DataFrame, None] = None) -> 'DatasplitToNumpyArray':
        return self
    
    def transform(self, X: Tuple[DataFrame, DataFrame, DataFrame, DataFrame]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        X_train, X_test, y_train, y_test = X
        return X_train.to_numpy(), X_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()

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