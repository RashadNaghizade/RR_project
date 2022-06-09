# Model and Visualize Mental Health in Tech 

## Table of contents
* [Main idea](#main-idea)
* [Description of the dataset](#description-of-the-dataset)
* [Data processing](#data-processing)
* [Visualisation and Insights](#Visualisation-and-Insights)
* [Prediction of mental health](#Prediction-of-mental-health)
* [Main conclusions](#main-conclusions)

## Main idea

The aim of this project is the reproduction and replication of the results of Andrada Oleanu from 2020 about menthal health in Tech industry.

Knowing the methodology used in the study, it will be checked whether the results remain the same when reproducing appropriate steps. Then it will be tested whether the outcome remains stable when extending the analysis by new observations comming from the survey we have conducted at our workplaces and comparing results on orginal data to new observations and combined data. Finally, the correctness of the conclusions will be also be checked with other initial assumptions and different types of models.

The topic of the original study was investigating the determinants of crime based on cross-sectional data from 2016 OSMI Mental Health in Tech Survey. The dependent variable in model was "has_current_mental_health_disorder", while independent variables include 62 variables such as sex, size of company, role at company, benefits and etc.

The results of the empirical study can provide guidance to managers in companies what is most importat to keep employees in good menthal health.

## Description of the dataset

The data for this study come mainly from USA and UK countries. Due to numerous outliers and not representative groups of responders in countries authors removed outliers and select observations from countries that have more than 30 responses. 

The selection of data in the final SWIID compilation resulted from the size of the databases - in the first place, the most extensive sets containing the most observations were used, and the gaps were supplemented with values from other sources. In the case of different values for the same observation, the value from a larger base was selected. The final version of the data set was created on the basis of three sources - the World Bank, Eurostat and SWIID.


## Data processing

The data contained a lot of missing values and the answers were not systematized. Therefore, the following actions were applied:

* rename columns
* sex columns & company size recoded
* removed outliers from age
* missing value listwise deletion (for variables where missing observations were more than half) and simple imputation
* column encoding
* country filtering (remained only with the ones with more than 30 responses)
* create tech column with flag 1/0


## Visualisation and Insights

In order to better understand the data and the relationships that occur between the answers, data visualizations were used regarding:

* country of survey participants
* size of company
* gender
* age
* mental health disorder by country
* approach to mental health of companies
* discussion menthal health at work
* approach to consequences of mental health 
* being open about mental health to friends and family¶

## Prediction of mental health

Ten machine learning models were used to classify the mental health of the survey participants. Then, using confiusion matrix, the best model was selected and by tuining its hyperparameters and the selection of explanatory variables author try to increase its precision.
Our contribution to the study was to examine whether the results will not change when we improve the hyperparameters of other models.

## Main conclusions

As a conclusion we want to mention main parts of the project.The original data had 1483 rows while our new data which we got from survey, had 50. If we consider that number of new observations compared to main one is less, it is acceptable that we are not to see extreme difference on results. Although to this, we can notice that it was affected by small percentage of changes which is normal in this case.However for better understanding, we uploaded the same 3 separate codes, but with different data, one with original, one with new observations and another with merged of both of these, with pictures of plots.
