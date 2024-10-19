# MLFinalProject
## Part 1
**Objective**: The Project Proposal is the first milestone in your final project journey. Its purpose is to provide a clear and concise outline of the machine learning problem or application you intend to explore. This milestone sets the foundation for your final project and requires you to articulate the significance of the chosen problem, describe the dataset you plan to use, and specify the machine learning algorithms you intend to explore.

**Tasks**:
1. **Team Members**:
    - Jesus Valdes
    - Pedro Villegas
    - Daniel Hernandez

2. **Select a Machine Learning Problem or Application**: Choose a machine learning problem or application that you find interesting and relevant. Clearly define the problem statement or the goal of your project.
    - The team has decided to build a machine learning software for the aerospace industry. In particular the airline industry.
    - One of the big concerns for airlines is safety. One of the biggest methods for airlines to keep a high standard of safety is through maintenance checks. 
    - The goal of this project is to develop an effective classification system to determine whether aircraft require maintenance based on a variety of operational and performance metrics.
    - This system aims to enhance safety, reduce downtime, and optimize maintenance scheduling by accurately identifying planes that are likely to need maintenance before issues arise.

4. **Dataset Description**: Provide a detailed description of the dataset you plan to use or intend to collect. Specify the source of the dataset (public sources like Kaggle, academic repositories, or collected personally, if applicable).
    - We are using the Turbofan Engine Degradation Simulation Data Set. The features on the dataset include time data, engine ID, 3 different operation settings, sensor measurements, and the estimated remaining life cycle of the engine.
    -	The dataset shows different data points, which are used for determining the predicted life cycle of an engine. 
    -	The dataset is obtained from kaggle and originated from NASA.

5. **Project Objectives**:Clearly outline the objectives of your project. What do you aim to achieve with your machine learning application?
    -	We aim to achieve accurate classification of aircraft that require maintenance.
    -	Optimize maintenance scheduling to minimize aircraft downtime.
    -	Lower maintenance costs by identifying potential issues early.
    -	Provide actionable insights to maintenance personnel through data analysis.
    -	The ultimate aim is to enhance aircraft reliability and safety through intelligent maintenance predictions, fostering a more efficient aviation ecosystem.

7. **Algorithms Specification**:
Specify the machine learning algorithms you intend to explore for solving the chosen problem. Provide a brief justification for your choice of algorithms.
    -	We plan to explore different classification algorithms, but mainly plan on using support vector machines(SVM) for robustness and capabilities.
    -	SVM has an overall advantage, due to its ability to handle large feature data effectively and it’s computationally efficient as well.
    -	Additionally, SVM has different kernels, which make it possible for handling the higher dimensional data. 
    -	The team had thought about incorporating the KNN  algorithm, but it isn’t decided yet due to its, in contrast, high computational cost.
  
## Part 2
- Deliver a 5-min demo that includes:
    - Project overview
    - Data preprocessing steps and any challenges encountered.
    - Preliminary results and any insights gained from the data analysis.
    - Plan for finalizing the project, including potential refinements and improvements.

**Tasks**:
create a power point of the following
1. write a project overview
2. download data and perform data preprocessing
    - write down any challanges found encountered 
3. analyze the data to gain any insights
    - write down any challanges found encountered
4. run initial test with the chosen model
    - write down any challanges found encountered
5. write down steps to finalize the project, including improvements, and refinements

**Challanges**
- preprocessing
    - dataset carried less columns than stated in the dataset description
        - dataset description originally stated the dataset to have 26 different sensor measurements, but in reality had only 21 sensor measurements
    - dataset was not meant for classification, a column for class was required to be added

## Citations
A. Saxena and K. Goebel (2008). “Turbofan Engine Degradation Simulation Data Set”, NASA Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA
