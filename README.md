# MLFinalProject
## How to use
For easy use of the code, follow the following steps:
- Data Preprocessing
    1. run Data_Preprocessing.py
        - the script runs: 
            - convert_to_csv.py
            - generate_classificaiton_targets.py
            - normalize_data.py
        - For more advance use of the code run files individually

- Data Analysis:
    1. run data_analysis.py
        - creates line graphs for sensors for engines
    - follow the terminal prompt instructions

- Model Training
    1. run train_SVM.py
  
## Part 2
- Deliver a 5-min demo that includes:
    - Project overview
    - Data preprocessing steps and any challenges encountered.
    - Preliminary results and any insights gained from the data analysis.
    - Plan for finalizing the project, including potential refinements and improvements.
### Tasks:
create a power point of the following:
1. write a project overview
2. download data and perform data preprocessing
    - write down any challanges found/encountered 
3. analyze the data to gain any insights
    - write down any challanges found/encountered
4. run initial test with the chosen model
    - write down any challanges found/encountered
5. present preliminary results
6. write down steps to finalize the project, including improvements, and refinements

### Project Overview
our goal for this project was to develop an effective classification system to determine whether an aircraft requires maintenance based on a variety of operational and performance metrics.

### Data Preprocessing
originally the dataset was composed of text files with values seperated by spaces. 
during this step:
- we converted the dataset files into csv files for easier use
- we normalized the data
- we generated classification targets for the data to best fit our problem statement

### Data Analysis Finding
during the data analysis step, it was found that sensor values drop after a certain amount of time, this is an indication that the engines require maintence before failure. This can be used to determine whether an engine needs maintenance by training the models off the sensors

It was also found that certain sensors are capable of being inversely proportionate to each other, although sensor to sensor influence has not been analyzed.

### Model Training
- for the SVM model 
    - the accuracy score resulted in 80%
    - we went with the following paramters:
        - trained all 21 sensors
        - a polynomial kernel
        - and a Regulation parameter of 1.0
    - during model training it was found that no matter the number of chosen features, the chosen kernel or the chosen regularization parameter the model would consistenly have an accuracy score of ~ 80%

### Challenges
- preprocessing
    - dataset carried less columns than stated in the dataset description
        - dataset description originally stated the dataset to have 26 different sensor measurements, but in reality had only 21 sensor measurements
    - dataset was not meant for classification, a column for class was required to be added
    - dataset required to be normalized
        - this was solved by using the mean and the standard deviation
- data analysis 
    - extra data processing is needed by setting a window during which an engine might need maintence, due to findings found during analysis.
- model training
    - SVM model
        - no issues

## Citations
A. Saxena and K. Goebel (2008). “Turbofan Engine Degradation Simulation Data Set”, NASA Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA
