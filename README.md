# House Prices - Advanced - Kaggle
- Kaggle competition link - https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview
- This is a Regression ML problem.
- The dataset contains 80 features of 1460 houses along with the prices at which these houses were sold mentioned under "SalePrice", which is our target feature. 

## Rank on Kaggle Public Leaderboard - 295 out of 4132

![Image](https://github.com/user-attachments/assets/ec02967d-486a-4d99-ace9-54d26814485b)

## Lowest RMSE achieved on Kaggle Leaderboard - 0.12045


## Problem Statement :
- Build a model to predict the sales price for each house in test dataset.
- Model's performance will be evaluated on Root-Mean-Squared-Error (RMSE) between the logarithm of the predicted value and the logarithm of the observed sales price.

## Tools used :
- Python - Coding Language
- Pandas - For Data Processing
- Numpy - For Arrays and Computations
- Sklearn - For ML Models and Metrics
- Optuna - For Hyperparameter Tuning
- Matplotlib/Seaborn - For Data Visualization

## Preprocessing:
#### Cleaning

![Image](https://github.com/user-attachments/assets/09c7bef4-f0e2-4758-af7b-2a62b980ad59)

- Fixed typo error in the name of categories of features "Exterior2nd", MSZoning", "Neighborhood" and "BldgType".  
- "GarageYrBlt" column sometimes has weird future values, like 2207, which are clearly data entry errors (probably meant 2007), so, to fix that if GarageYrBlt > 2010 (the dataset is from 2010), it’s invalid and in those cases, we replace it with the house’s "YearBuilt" instead (Assumption: garage was built same time as house).
- Features "1stFlrSF", "2ndFlrSF", "3SsnPorch renamed to "FirstFlrSF", "SecondFlrSF" and "Threeseasonporch" for easier readability and interpretability. 
#### Imputation
- By specifying exact order and using "CategoricalDtype", Ordinal Categorical features prepared for encoding and defining features as an ordered categorical type in Pandas.
- Mode Imputation done for all categorical features.
- Median Imputation done for all numerical features.
#### Encoding
- Label Encoding is applied to the dataset.
- On applying Target Encoding using "MEstimateEncoder", errors increased so, only Label Encoded dataset used for final submission.

## Feature Engineering and Outliers:
- Dropped features having MI scores less than or equal to "0".
- Created feature "LivLotRatio" to indicate ratio of living area and lot area.
- Using "FirstFlrSF" (sq. footage of 1st floor), "SecondFlrSF" and "TotRmsAbvGrd", created feature "Spaciousness" denoting average sq. footage per room (a measure of how roomy each room is).
- Created interaction features like that of One-Hot encoded "BldgType" and "GrLivArea", that of Label encoded "OverallCond" and Label encoded "OverallQual" creating "ResultCond" and that of "TotalBsmtSF" and Label encoded "BsmtQual" creating "BsmtWeightage".
- Linearized area by taking square root of all the area features.
- Created feature showing total count of "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "Threeseasonporch" and "ScreenPorch" in houses.
- Extracted numeric class data from "MSSubClass" by splitting it at underscore and naming the feature as "MSClass".
- Created "MedNhbdArea" by computing median of "GrLivArea" for each "Neighborhood".
- Computed deviation of "GrLivArea" from the neighborhood’s median house size "MedNhbdArea" as feature "VarFromMed".
- Using "KMeans" clustering done to obtain feature "Cluster" (containing clusters to which each samples belong to) and a dataframe containing distance from each cluster for each sample.
- Using "Principal Component Analysis (PCA)" obtained features that captured most variance and created "Feature 1", "Feature 2" and "Feature 3" using few of the most variance capturing features "GrLivArea", "TotalBsmtSF", "YearRemodAdd" and "GarageArea".
- Created binary feature "Outlier" indicating presence of "Edwards" in "Neighborhood" feature and "Partial" in "SaleCondition" feature.

## Modeling:
- "XGBoost Regressor" is used for modeling.
- "Optuna" is used for Hyperparameter tuning.

## Final Submission:
- Best Hyperparameters obtained using "Optuna" used to generate final predictions using "XGBoost Regressor".
- Lowest RMSE of 0.12045 has been achieved on Kaggle's Public Leaderboard leading to rank of 295 out of 4132 submissions.
