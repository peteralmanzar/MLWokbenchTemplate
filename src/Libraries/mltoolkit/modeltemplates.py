from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Flatten, Conv1D, MaxPooling1D, Conv2D, MaxPooling2D

def GetModelTemplateLSTM(numberOfSteps: int, numberOfFeatures: int):
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

def GetModelTemplateMLP(numberOfFeatures: int):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(numberOfFeatures,)),
        Dense(64, activation='relu'),
        Dense(1)
    ])

    return model

def GetModelTemplate1DCNN(numberOfSteps: int, numberOfFeatures: int):
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

def GetModelTemplate2DCNN(image_height: int, image_width: int):
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