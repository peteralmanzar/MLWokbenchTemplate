from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Conv2D, Dense, Dropout, Flatten, LSTM, MaxPooling1D, MaxPooling2D

def GetModelTemplateMLPRegression(numberOfFeatures: int):
    '''
    Returns a template for a Multi-Layer Perceptron (MLP) model for regression tasks.
    '''
    model = Sequential([
        Dense(64, activation='relu', input_shape=(numberOfFeatures,)),
        Dense(64, activation='relu'),
        Dense(1)
    ])

    return model

def GetModelTemplateMLPClassification(numberOfFeatures: int):
    '''
    Returns a template for a Multi-Layer Perceptron (MLP) model for classification tasks.
    '''
    model = Sequential([
        Dense(64, activation='relu', input_shape=(numberOfFeatures,)),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    return model

def GetModelTemplateMLPClassification(numberOfFeatures: int, num_classes: int):
    '''
    Returns a template for a Multi-Layer Perceptron (MLP) model for classification tasks.
    '''
    model = Sequential([
        Dense(64, activation='relu', input_shape=(numberOfFeatures,)),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    return model 

def GetModelTemplateLSTM(numberOfSteps: int, numberOfFeatures: int):
    '''
    Returns a template for a Long Short-Term Memory (LSTM) model. Ideal for time series forecasting.
    '''
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(numberOfSteps, numberOfFeatures)),
        Dropout(0.2),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=True),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1)
    ])

    return model

def GetModelTemplate1DCNN(numberOfSteps: int, numberOfFeatures: int):
    '''
    Returns a template for a 1D Convolutional Neural Network (CNN) model. Ideal for time series forecasting.
    '''
    model = Sequential([
        Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(numberOfSteps, numberOfFeatures)),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=64, kernel_size=2, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(50, activation='relu'),
        Dense(1)
    ])

    return model

def GetModelTemplate2DCNN(image_height: int, image_width: int, num_classes: int):
    '''
    Returns a template for a 2D Convolutional Neural Network (CNN) model. Ideal for image classification tasks.
    '''
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(image_height, image_width, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    return model