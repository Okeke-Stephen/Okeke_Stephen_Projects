# Predicting-customer-buying-behaviour

## Introduction
In this repository, I've put together a neat machine learning model that predicts if a customer will make a booking. The cherry on top? I've used a RandomForest model as it's straightforward to interpret and awesome at ranking feature importance.
Here's a glimpse of my process:

## Explore and Prepare the Dataset
I started with the "Getting Started" Jupyter Notebook to familiarize myself with the dataset. I explored each column, generated a statistical overview of the data, and sought out any missing data.
## Preparing the Data
Next, I prepped the dataset for the RandomForest model. This step involved filling in missing values, translating categorical variables into a language that my model could comprehend, and scaling continuous variables. Here, I also brought my creativity into play. I engineered some new features that I believed could boost the model's performance.

## Train a Machine Learning Model
After prepping, I split the data into a training set and a testing set. This allowed me to train my RandomForest model on known data and then test its performance on unseen data. The RandomForest model was my star performer here. It's a versatile, user-friendly algorithm that consistently delivers excellent results, even without hyper-parameter tuning. But I didn't stop there. I further refined my model by tuning its hyperparameters using techniques like GridSearch and RandomSearch.

## Evaluate Model and Present Findings
Finally, I put my model to the ultimate test. I evaluated its performance using cross-validation, ensuring that the results weren't just due to a particular arrangement of data. In presenting my findings, I included evaluation metrics such as precision, recall, f1-score, and accuracy to provide a well-rounded view of the model's performance. I also used RandomForest's feature importance visualization to highlight the key variables influencing the predictions.

I hope you enjoy exploring this project as much as I did building it!
