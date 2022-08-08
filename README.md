# Predicting Log Error for Zillow


## About the Project
### Project Goals

My goal with this project is to identify Zillow's key drivers of logerror and to provide insight into why and how these factors are producing certain log errors. With this information and the following recommendations, our organization can work together to improve business processes and procedures in order to more accurately predict home values and log error moving forward.


### Project Description

At Zillow, the ability to predict logerror is essential for measuring the error of a home value prediction within this database.

In order to more accurately predict log error, we will analyze the attributes (features) of homes within a predetermined set of data. This dataset includes Single Family Properties that had a transaction during 2017.
We will then develop models for predicting log error based on these attributes and provide recommendations and predictions to Zillow for improving prediction of log error moving forward.


### Initial Questions

#### 1. Does a higher number of bedrooms increase logerror?

#### 2. Does a higher number of bathrooms increase logerror?

#### 3. Do more garage spaces increase logerror?

#### 4. Is logerror significantly different for properties in LA County vs Orange County vs Ventura County?

#### 5. Does a higher square footage increase home value?



### Data Dictionary

| Variable      | Meaning |
| ----------- | ----------- |
| logerror      | The measured log error of a home       |
| home_value      | The total tax assessed value of the parcel       |
| bedrooms   | The total number of bedrooms in a home        |
| bathrooms      | The total number of bathrooms in a home       |
| garage_spaces      | The total number of car slots in a garage       |
| year_built      | The year the home was built       |
| age      | The age of the home       |
| location      | Location of a home by county      |
| sq_ft      | The total square feet of a home       |
| lot_sq_ft      | The total square feet of a property lot       |
| latitude   | Location using the latitudenal metric        |
| longitude      | Location using the longitudenal metric       |
| bath_bed_ratio      | Ratio of bathrooms to bedrooms of a home       |


### Steps to Reproduce

1. You will need an env.py file that contains the hostname, username and password of the mySQL database that contains the zillow dataset. Store that env file locally in the repository.
2. Clone my repo (including the acquire_telco.py, prepare.py) 
   (confirm .gitignore is hiding your env.py file)
3. To acquire the zillow data, I used the zillow_db in our mySQL server. I selected all columns from the properties_2017 table. I then joined this table with the propertylandusetype and predictions_2017 in order to narrow our data to reflect Single Family Properties that had a transaction during 2017. 
4. Libraries used are pandas, matplotlib, seaborn, numpy, sklearn, scipy, and model. A full list of modules with specific tools are provided in my Full Report.
5. Following these steps, you should return the exact dataset I used to in my report.


### The Plan
Below, I walk through all stages of my pipeline and process.

#### Wrangle
##### Modules (acquire.py + prepare.py)

1. Test acquire function
2. Add to acquire.py module
3. Write and test function to clean data
4. Add to prepare.py module
5. Write and test function to split data
6. Add to prepare.py module

#### Explore 
##### Modules (explore.py)

1. Ask 5 distinct questions of our data \
  a. Does a higher number of bedrooms increase home value? \
  b. Does a higher number of bathrooms increase home value? \
  c. Do more garage spaces increase home value? \
  d. Does location by county affect home value? \
  e. Does a higher square footage increase home value? \
2. Explore these questions through visualizations, calling explore.py as needed \
  a. Barplots are used primarily due to our features being categorical variables \
  b. For our continuous variable, lmplots with line of best fit is used \
  c. These plots illustrate correlation of our chosen features with home value \
3. Statistical Testing is conducted on all relevant features to determine statistical significance \
4. Summary includes key takeaways from all features explored \

##### Clustering Exploration

1. Goal is 3 distinct clusters \
  a. Does bedroom, bathroom, and garage space count affect log error?
  b. Does location, latitude, and longitude affect log error?
  c. Do sqft, lot_sq_ft, and bath_bed_ratio affect log error?
2. Multiple questions will be asked for each cluster, exploring through visualizations, calling explore.py as needed \
3. Summary includes key takeaways from all features explored \

#### Modeling and Evaluate
##### Modules (model.py)

1. Select Evaluation Metric: Correlation, namely RMSE
2. Scale the data utilizing our model.py scaling function
3. Evaluate a Baseline: 272,118 (error in dollars)
4. Develop 3 distinct models
    a. Linear Regression
    b. Lasso Lars
    c. TweedieRegressor
5. Evaluate on Train and then on Validate (for promising feature sets)
6. Once a top performing model is selected, evaluate on test dataset



### Conclusion

#### Summary

In seeking solutions to more accurately predict log error for Zillow, we have explored a multiplicity of factors in the dataset that affect log error. We have shown that some potential primary drivers of log error are :

- The number of bedrooms in a home
- The number of bathrooms in a home 
- The number of garage spaces in a home
- The location of a home by county
- The square footage of a home

In addition to these more self-evident factors, we have created meaningful clusters using Kmeans in order to create better predictions. These clusters include:

- The number of bedrooms and bathrooms, and the age of a home
- Latitude, longitude, and location by county

The correlation of these features and our newly created clusters with log error, combined within our analysis and models, expresses confidence in the validity of our findings. We have created models that perform slightly better than our baseline of 0.169.

Having fit the best performing model to our train, validate, and test datasets, we expect this model to perform 0.14% better than our baseline in the future on data it has not seen, given no major changes to our data source.

#### Recommendations

There are a number of recommendations that can be offered based on the above analysis. These suggestions are tied to the relative lack in performance within our primary drivers and clusters when predicting logerror:

1. I recommend that Zillow considers collecting more data on location-driven features such as proximity of local schools, emergency services such as police and fire stations, and local parks and recreational areas.
2. Based on our exploration of clusters that affect log error, I recommend Zillow further investigate and extrapolate on clustering techniques in order to better predict log error.

#### Next Steps

Considering the overall lack in effectiveness of our best-performing model, there is certainly room for improvement and optimization. \
If given more time to pursue a better results, I would begin by conducting further exploration and analysis of the clusters I created for our dataset. This could include:
- Exploring census data further in order to better identify specific neighborhoods instead of simply analyzing at the county level.
- Re-calibrating my cluster based on home size by exploring more appropriate features to include.

By optimizing our dataset to include the above categories, I believe we could increase the correlation of our feature set with log error and improve model prediction accuracy.

