---
layout: post
title:  "Solar Energy Prediction - Aneo"
date:   2024-10-10 12:07:13 +0200
categories: jekyll update
---

The project is based of a competition on forecasting solar energy prediction hosted by Aneo and NTNU. The objective is to predict hourly solar energy production for the next day across several locations.

*Erlend Lokna and myself ranked 1st out of 100 in the Kaggle competition.*

## Problem description

Based on historical metrological and predicted metrological data, produce the best performing prediction model for next day hourly solar energy production in term of MAE.
### Dataset 

This dataset provides data for evaluating solar production day-ahead forecasting methods. 
The dataset contains metrological data measured in three separate locations (A, B, C), corresponding to office buildings with solar panels installed. Feature descriptions can be found [here](https://www.meteomatics.com/en/api/available-parameters/alphabetic-list/).

The dataset contains historical metrological data, with corresponding power production, serving as training data. The test data is *estimated*, which makes evaluation equivalent to forecasting methods that are used in production.

## Data Processing and Analysis
### EDA

#### Visualizing and examining the data
![target](/assets/target.png)

#### Missing values - NaNs
![nan](/assets/nan.png)

#### Distributional Differences Across Locations
![dirct radiation](/assets/dir_rad.png)

Crunched the numbers to confirm that the differences are quite small. Calculated the MAE of the features of A, B and C, and the only features with values higher that 0,5 was for the MAE has joule as unit, which is a very fine measurement unit for energy. The MAE of 21345 J is approximately 5 watt hours, which is enough energy to power a LED light for somewhere around an hour.

We conclude that the distributional difference of the data for location A, B and C is very small. Exeptionally small between A and B.

#### Covariance Analysis - Data redundancies
![covariance](/assets/cov_analysis.png)
### Data Cleaning

**Drop nans**
We worked solely with the non NaN columns, as data was abundant.

**Drop features**
From the initial investigations of the data we saw that several of the features contained very little information. We attempted a lot of different subsets, removing most variables having to do with snow as these were basically constant.

**Drop bad data**
From the plot of Y across the different locations we saw that there were several spots of constant values. We use `drop_bad_data` to identify the indexes in Y which are part of 12 consecutive constant hours which do not have the value 0. Then we remove these and the corresponding X values.
### Feature Engineering

**Resampling X to hour**:
The metrological data (X) came in 15 minutes intervals, while the target data (Y) came in 1 hour intervals. We therefore resampled the X values to match Y by aggregating the mean of the X data over the last hour.

**Added features**:
*Location*: Categorical feature with value A, B or C depending on where the data comes from.
Idea: For the model to learn the intricate differences between the three locations.

*Type*: Categorical feature with value 0 (= Estimated) or 1 (= Observed) indicating if the data is observed or estimated.
Idea: For the model to take the distributional difference between estimated and observed into account.

**Making Categorical**
Several features of the data are categorical, but pandas fails to pick it up automatically. We therefore explicitly made the categorical features of type category. 


**Scale depending on location**
By the clear distributional similarities in the metrological data across locations, we got the idea of scaling the Y values for each of the locations by a factor, such that the scale across locations would stay similar. Then, after fitting the model and predicting values we scale back using the same factors. Experimenting with different scale values lead us to leaving A as it is, scaling B with $k_b = 5$ and scaling C with $k_c = 6$. The result was an improvement in prediction accuracy. We hypothesize that this is a result of the model not needing to determine the scale factor itself, but rather focus on more intricate differences.

**Cyclic time encoding**
The data is clearly of a cyclic nature, since it weather data that spans over several years, and seasons exist. The models do not like the human date format for time. In order to keep the cyclic patterns and the passage of time in the data we encoded the months and hours as points on a circle by taking a sine and cosine transformation of the hour and month as described in `cyclic_time_encoding()` above.

##  Model Fitting and Tuning

**Predicted Values <5 set to 0**
To better control random fluctuations in areas where the predictions should be constant, we dropped sufficiently low values to 0. Through trial and error we set this value to 5. 

### Model 1: CatBoost
This has been our main model for the entire project. Almost all our experiments has been with CatBoost. Fine tuning the parameters and tweaking the data. The reason for this is that the model provided good result out of the box, and that before we were able to harness the information in the data, different models would probably not provide substantially better results. Sticking to one model for a long time made it easier to control the experiments.

We attempted to use Optuna for hyper parameter optimization. But ended up with better results by some trial and error.

#### One big vs three small

We tested one big training data set with all data from A, B and C together, and then training one model, as well as three separate models, one for each location. The big model provided better results throughout, and less computation time, than the small ones. Also, when we introduced `y_scale()` to the big model was far better.

#### Summer month model

As the task of this project was to predict the solar power generation in the summer months, we thought maybe that a model trained solely on these months would perform ok. Either as a single model or part of an ensemble. It did not provide any great results.
### Model 2: AutoGluon

We fit a model for each location, holding back $50 \%$ of the estimated training data in order to tune it later in order to better take the distributional difference of the observed and estimated data into account.