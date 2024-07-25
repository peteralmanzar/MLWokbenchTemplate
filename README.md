# Workbench
## Index
- Directories
    - src
        - Data
            - Raw
            - Training
        - Libraries
        - Persistence
            - Models
            - Pipelines
- Components
    - Study
    - Libraries
        - parameters
        - workbenchlib
        - mltoolkit
            - datatransformers
            - modeltemplates
            - modeltunning
            - persistence
- Where to Start

## Directories
**src**
: root directory

**Data**
: contains all data

**Raw**
: contains all unprocessed data

**Training**
: contains data used for training a model

**Libraries**
: contains all code used by the study and presentation notebooks

**Persistence**
: contains saved objects

**Models**
: saved trained models

**Pipelines** 
: saved fitted pipelines

## Components
**Study**
: notebook used by the analyst to study and process the data as well as train and evaluate a model.

### Libraries
directory where all code heavy logic resides 

**parameter**
: functions and variables that must be edited and assigned values

**workbenchlib**
library of functions used withing the study and presentation

### mltoolkit
suite of tools commonly used in machine learning.

**datatransformers**
: contains data transformers used for processing data

**modeltemplates**
: contains python code to get pre built model templates

**modeltunning**
: contains python code to facilitate model hyper parameter tunning

**persistence**
: contains python code to facilitate saving and loading

## Where to Start
if your data is in a file such as a `.csv` make sure the file is under the `Data>Raw` directory.

You want to start by running the `Study_Guide.ipynb`. You can create a copy or rename it(i.e.: `Study.ipynb`). The guide contains the bare minimum to get started with trainig an Machine learning model. Run the first few cells in order to load the data and perform an EDA. This should get you an idea of what transformers are needed to preprocess your data and possibly what model to use.

To get the rest of `Study_Guide.ipynb` working you need to define some functions in the `parameter.py` file. 
Provide a pipeline by defining `getPreprocessingPipeline`. See `datatransformers.py` for a list of custom transformers that do common data preprocessing tasks. 
Provide a model by defining `getModel`. See `modeltemplates.py` for a list of pre built model templates.
Provide a target for the model to predict by defining `getTarget`.