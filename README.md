###  **Bank Customer Action Prediction**

###  Author : Umidjon Sattorov, Machine Learning Engineer, successful graduate of the SkillBox platform, and student at Mohirdev platform

**Overview**

This project focuses on predicting whether a customer of a bank performs specific actions when they visit the bank's website. The project involved comprehensive data analysis to answer financial questions provided by the dataset and the development of an advanced machine learning model. After extensive training and evaluation of various algorithms, the final model achieved an impressive ROC AUC score of 0.9, demonstrating exceptional predictive performance.

**Features**

High Accuracy: Achieved a ROC AUC score of 0.9, indicating strong model performance.
Multiple Algorithms: Evaluated numerous machine learning algorithms to identify the best-performing model.
Customer Action Prediction: Predicts whether a customer performs specific actions on the bank's website based on various features.
Comprehensive Analysis: Includes detailed analysis and preprocessing steps to ensure data quality and model robustness.

**Technologies Used**

Python: Core programming language for data processing and model development.
pandas: Data manipulation and analysis.
scikit-learn: Machine learning library for model training and evaluation.
FastAPI: Framework for creating APIs to deploy the model.

**Results**

The model’s high ROC AUC score of 0.9 highlights its ability to accurately predict customer actions, making it a valuable tool for understanding customer behavior and improving the bank's online services. The final model used in this project is an MLPClassifier with optimal hyperparameters. Here you can see the models performance on the server : 

![First_prediction](https://github.com/user-attachments/assets/d3d9fd0c-91fb-4fae-9fed-279f58f22243)

![Second_prediction](https://github.com/user-attachments/assets/30545f76-c178-4f80-8f35-7f6755f34357)

![Third_Prediction](https://github.com/user-attachments/assets/52f0d8f2-f957-46f5-bc5d-80abb6144d9b)

![Fourth_prediction](https://github.com/user-attachments/assets/22e2b827-759b-4556-81a1-f7422f92ac66)

The graph below is measured after model is trained(Models roc_auc score is measured only for test dataset): 

![image](https://github.com/user-attachments/assets/571ce5d8-a026-428f-b81c-d88eb2abac99)

Image below can demonstrate the structure of the pipeline, but pipeline which picture illustrated is the pipeline created with Random Forests algorithm. Actual algorithm which is used in final pipeline is MLP classifier.

![image](https://github.com/user-attachments/assets/340e3624-6e0d-4af0-bd26-6f1d62ba8447)

**File Structure and Descriptions**

individual_analysis_2.ipynb: Analysis of the data for financial and marketing questions.

individual_analysis_3.ipynb: Continued analysis focusing on additional financial metrics and insights.

individual_analysis_4.ipynb: Final analysis addressing remaining questions and summarizing findings.

main.ipynb: Initial model development and evaluation using various machine learning algorithms.

renewed.ipynb: Refinement of the model with optimal hyperparameters and final evaluation.

pipeline: Contains the complete pipeline for preprocessing steps and the final model, ensuring streamlined and reproducible workflows.

FastAPI server: Implementation of a FastAPI server to test the model in a server application, which worked seamlessly and returned outstanding results.

Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or suggestions.
