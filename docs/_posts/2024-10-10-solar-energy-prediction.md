---
layout: post
title:  "Solar Energy Prediction - Aneo"
date:   2024-10-10 12:07:13 +0200
categories: jekyll update
---

This project was developed as part of a competition on forecasting solar energy production, hosted by Aneo and NTNU. The objective was to predict hourly solar energy production for the next day across several locations using historical and forecasted meteorological data.

**Achievement:** Together with Erlend Lokna, I secured 1st place out of 100 participating teams in the Kaggle competition associated with this challenge.

## Problem description
The goal was to create a high-performing predictive model for next-day hourly solar energy production, measured in terms of Mean Absolute Error (MAE). The model needed to leverage both historical meteorological data and forecasted meteorological data to make accurate predictions across multiple locations

### Dataset 

The dataset provided for this project was specifically designed to evaluate day-ahead solar production forecasting methods. It includes:

- Meteorological Data: Historical and forecasted data measured at 15-minute intervals.

- Power Production Data: Corresponding solar energy production measured hourly.
- Locations: Data from three separate locations—A, B, and C—which are office buildings equipped with solar panels.

Feature descriptions and parameters can be found in detail [here](https://www.meteomatics.com/en/api/available-parameters/alphabetic-list/).

The test data is estimated, simulating real-world forecasting conditions where future values are not observed but predicted.

## Data Processing and Analysis
### EDA

#### Visualizing and examining the data

To understand the underlying patterns and distributions, we began by visualizing the target variable—hourly solar energy production—for each location.
![target](/assets/target.png)

It is immediately evident that the data is noisy, has missing values, and faulty measurements. Additionally we observe that the scale of energy prediction at location A is substantially larger than at B and C. The major seasonal trends are similar, which indicate that the locations are not located very far apart. This is positive for possible transfer learning. 


#### Missing Values (NaNs)
An initial inspection revealed the presence of missing values (NaNs) in some features.

![nan](/assets/nan.png)

We observed that while some features had missing values, the overall dataset was sufficiently large, allowing us to proceed without extensive data imputation.

#### Distributional Differences Across Locations
We examined the distributions of key meteorological features, such as direct radiation, across the three locations.

![dirct radiation](/assets/dir_rad.png)

To quantify the distributional differences, we calculated the MAE for each feature between locations A, B, and C. The differences were minimal, with only features measured in joules showing MAE values above 0.5. Specifically, the MAE of 21 345 J (approximately 5 watt-hours) is relatively insignificant, equating to the energy needed to power an LED light for about an hour.

The meteorological data across the three locations have very similar distributions, particularly between locations A and B. This suggests that a single model could potentially generalize well across all locations.

#### Covariance Analysis - Identifying Data redundancies

To detect potential redundancies among features, we conducted a covariance analysis.

![covariance](/assets/cov_analysis.png)

This analysis highlighted highly correlated features, indicating that some variables could be redundant. Identifying and possibly removing these features can simplify the model.

### Data Cleaning

**Drop nans**
Given the abundance of data, we chose to work exclusively with columns that did not contain NaN values. This decision minimized the complexity of data preprocessing and avoided potential biases from imputation methods.

**Feature Reduction**

Our initial investigations revealed that some features provided little to no valuable information for the prediction task. For example
- Snow-Related Variables: These variables were largely constant throughout the dataset, likely due to the geographic locations and time frames involved.
- Low-Variance Features: Features with minimal variance offer limited predictive power.
We systematically removed these low-variance and uninformative features to streamline the dataset and improve model efficiency.


**Removing Erroneous Data**
Upon plotting the target variable across different locations, we noticed periods where the energy production values remained constant for extended durations, which was unusual, especially when the values were non-zero. To address this we removed these data points along with their corresponding feature values.
This step ensured that anomalous data did not adversely affect model training.

### Feature Engineering

**Resampling Meteorological Data to Hourly Intervals**:

The meteorological data (X) was provided in 15-minute intervals, while the target variable (Y) was recorded hourly. We resampled the meteorological data by calculating the mean over each hour. This ensured that each target value corresponded directly with the aggregated features for that hour.

**Adding New Features**:

*Location*: We introduced a categorical feature with value A, B or C depending on where the data comes from. The aim was to assist the model learn location-specific patterns.

*Type*: Categorical feature with value 0 (= Estimated) or 1 (= Observed) indicating if the data is observed or estimated.
The purpose was for the model to take distributional difference between estimated and observed into account.

These additions allowed the model to adjust predictions based on the origin and nature of the data.


**Scaling Target Variable Based on Location**

Given the minimal distributional differences in meteorological data across locations but potential differences in solar energy production scales, we hypothesized that scaling the target variable (Y) for each location could improve model performance.

Scaling Factors:
- Location A: No scaling (k_A = 1).
- Location B: Scaled by a factor of k_B= 5.
- Location C: Scaled by a factor of k_C = 6.
Process:
- Before Training: Multiply the target variable by the scaling factor corresponding to its location.
- After Prediction: Divide the predicted values by the same scaling factor to return to the original scale.

Aligning the scales across locations allows the model to focus on learning patterns rather than adjusting for scale differences. Experimentation showed that this scaling improved prediction accuracy, likely by simplifying the learning task for the model.


**Cyclic time encoding**

Solar energy production exhibits strong cyclical patterns related to solar irradiance, with daily and seasonal variations. 

We transformed the day and month data by applying a sine/cosine transform, essentially mapping the data on a circle. This encodes cyclical nature of time, allowing the model to understand that, for example, midnight and 1 AM are adjacent times. It removes unnatural discontinuities by preventing the model from interpreting the highest and lowest numerical values as being far apart.


##  Model Fitting and Tuning

**Predicted Values <5 set to 0**

To address random fluctuations and improve the stability of our predictions we set any predicted values below 5 to 0. Solar panels do not produce significant energy below certain light levels, so low predictions are likely noise. The threshold value of 5 was chosen through trial and error, balancing the need to eliminate noise without discarding valid low-energy outputs.
 

### Model 1: CatBoost
CatBoost served as our primary model throughout the project due to its strong out-of-the-box performance. Its ability to natively handle categorical features without explicit encoding, combined with resistance to overfitting and robustness against noisy data, made it an ideal choice for our needs. We concentrated almost all our experimentation on CatBoost, fine-tuning parameters and adjusting the data to optimize results. This focused approach allowed us to harness the information in the data more effectively and maintain better control over our experiments.

While we initially explored hyperparameter optimization using Optuna, we ultimately achieved superior results through manual trial and error. This hands-on method enabled us to make nuanced adjustments that automated processes might overlook.

#### Training Strategies

We investigated different training strategies to enhance model performance. One approach involved combining all data from locations A, B, and C into a single, large training dataset to train one comprehensive model. Alternatively, we trained three separate models, one for each location. The combined model consistently outperformed the individual models, delivering better results with less computation time. The performance gap widened further when we introduced scaling of the target values; the combined model adapted more effectively to the scaling adjustments than the separate models.

#### Summer month model
Considering that the project's objective was to predict solar power generation during the summer months, we hypothesized that a model trained exclusively on summer data might yield better performance. We tested this idea by creating a model trained solely on data from the summer months, both as a standalone model and as part of an ensemble. However, this approach did not produce significant improvements, suggesting that including data from all seasons provided the model with valuable context that enhanced its predictive capabilities.

### Model 2: AutoGluon
As an alternative to CatBoost, we experimented with AutoGluon to develop models for each location individually. In this process, we withheld 50% of the estimated training data for later tuning, aiming to account more effectively for the distributional differences between observed and estimated data. The intention was to refine the models to better capture location-specific patterns and discrepancies. Despite these efforts, the AutoGluon models did not surpass the performance of the CatBoost model, reaffirming our decision to prioritize CatBoost for this project.

## Conclusion

By employing meticulous data cleaning, thoughtful feature engineering, and strategic model selection and tuning, we successfully developed a predictive model for next-day hourly solar energy production.