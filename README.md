# Predicting Log Error from Zillow's Zestimate

This project is an extension of a previous project. View the github of the previous project here: [**Predicting Tax Assessed Home Values**](https://github.com/Estimating-Home-Value/regression-project)

## Description
> For this project you will continue working with the zillow dataset. Continue to use the 2017 properties and predictions data for single unit / single family homes.

> In addition to continuing work on your previous project, you should incorporate clustering methodologies on this project.

> Your audience for this project is a data science team. The presentation will consist of a notebook demo of the discoveries you made and work you have done related to uncovering what the drivers of the error in the zestimate is.

## Goals
1. Identify significant drivers of log error for single unit properties in 2017.
2. Utilize clustering methodologies to develop derived features. 
3. Develop a model that predicts log error with a lower root mean squared error than a featureless baseline model.
4. Discuss findings and provide recommendations for next steps.

## Deliverables
1. A clearly named final notebook. This notebook will be what you present and should contain plenty of markdown documentation and cleaned up code.
2. A README that explains what the project is, how to reproduce you work, and your notes from project planning.
3. A Python module or modules that automate the data acquisistion and preparation process. These modules should be imported and used in your final notebook.

## Additional Project Requirements
- Data Acquisition: Data is collected from the codeup cloud database with an appropriate SQL query
- Data Prep: Column data types are appropriate for the data they contain
- Data Prep: Missing values are investigated and handled
- Data Prep: Outliers are investigated and handled
- Exploration: the interaction between independent variables and the target variable is explored using visualization and statistical testing
- Exploration: Clustering is used to explore the data. A conclusion, supported by statistical testing and visualization, is drawn on whether or not the clusters are helpful/useful. At least 3 combinations of features for clustering should be tried.
- Modeling: At least 4 different models are created and their performance is compared. One model is the distinct combination of algorithm, hyperparameters, and features.
- Best practices on data splitting are followed
- The final notebook has a good title and the documentation within is sufficiently explanatory and of high quality
- Decisions and judgment calls are made and explained/documented
- All python code is of high quality

## Key Findings and Takeways:
1. Log error is normally distributed around a mean of 0.016
2. Larger more valuable homes appear to be increasing the mean log error. This assertion is supported by statistical tests.
3. A Linear Regression model with Polynomial 3rd Degree Features selected by Recursive Feature Elimination outperformed a baseline featureless model and consists of features based on the following:
    * Square footage of the main property structure
    * Tax assessed property value and assessed tax amounts
    * Location
4. While the model outperformed baseline, the improvement was marginal, and a more robust algorithm is recommended for future investigation.

## Data Dictionary
| Column | Description |
| --- | ---|
| id | Autoincremented unique index id for each record |
| parcelid | Unique identifier for parcels (lots) |
| bathroomcnt | Number of bathrroms in home including fractional bathrooms |
| bedroomcnt | Number of bedrooms in home |
| buildingqualitytypeid | Overall assessment of condition of the building from best (lowest) to worst (highest) |
| calculatedbathnbr | Number of bathrooms in home including fractional bathrooms. Appears to be redundant with bathroomcnt |
| calculatedfinishedsquarefeet | Calculated total finished living area of the home |
| finishedsquarefeet12| Finished living area | 
| fips | Federal Information Processing System codes used to identify unique geographical areas, converted to county names during data prepartion | 
| fullbathcnt | Number of full bathrooms (sink, shower + bathtub, and toilet) |
| heatingorsystemtypeid | Numeric value representing a heating system type. Matches heatingorsystemdesc | 
| latitude | The latitude of the property
| longitude | The longitude of the property |
| lotsizesquarefeet| Area of the lot in square feet |
| propertycountylandusecode | County land use code i.e. it's zoning at the county level |
| propertylandusetypeid |  Type of land use the property is zoned for |
| propertyzoningdesc | Description of the allowed land uses (zoning) for that property | 
| rawcensustractandblock | Government id for each property linked to geographic location |
| regionidcity | City in which the property is located (if any) |
| regionidcounty | County in which the property is located |
| regionidzip | Zip code in which the property is located | 
| roomcnt | Total number of rooms in the principal residence |
| unitcnt | Number representing the number of units on the property |
| yearbuilt | The year the principal residence was built |
| structuretaxvaluedollarcnt | The tax assessed value of only the property structure in USD |
| taxvaluedollarcnt | The tax accessed value of the property in USD | 
| assessmentyear | Year that the tax value was assessed |
| landtaxvaluedollarcnt | The tax assessed value of only the land area for the parcel |
| taxamount | The total property tax assessed for that assessment year |
| censustractandblock | Redundant with rawcensustractandblock |
| logerror | log error of Zillow's Zestimate model |
| transactiondate | Four digit year, two digit month, two digit date | 
| heatingorsystemdesc | Description of the type of heating system on the property |
| propertylandusedesc | Description of the type of property | 

Additional columns were present in the zillow database but had greater than 50% null values and were dropped during initial consideration. 

## Data Acquisition and Validation
The data was acquired through the acquire.prepare_zillow function that performed the following:
1. Retrieved 73,695 records from the Codeup Zillow database using a SQL query with the following requirements:
    1. Transaction Date between 2017-01-01 and 2017-12-31
    2. Property classified as one of the following Types:
        * Single Family Residential
        * Rural Residence
        * Mobile Home
        * Townhouse
        * Condominium
        * Row House
        * Planned Unit Development
        * Bungalow
        * Patio Home
        * Inferred Single Family Residential
    3. Latitude and Longitude data is not null.
    4. For properties with more than one transaction date, only the record containing the latest transaction dates were returned.
2. Records containing 0 bathrooms, 0 bedrooms, or 0 square footage were dropped.
3. Records with a unitcnt greater than 1 were dropped.
3. Duplicate records were dropped. These entries may represent "back-to-back" closings on the same day between three parties.
4. Columns with greater than 50% missing values were dropped
5. Records missing more than 25% of their values were dropped
6. The following columns were dropped:
    * **id** - The information contained in this column is a unique identifier within the SQL database. The data has no meaning specific to the unique characteristics of the property. 
    * **parcelid** - Similar to id, this data represents a unique identifier for the SQL database. 
    * **assessmentyear** - Every entry for the data in question has the same value (2016) and will provide no insight during modeling.
    * **calculatedbathnbr** - Contains a large number of null values. This feature may be derived from some combination of full bathrooms, half bathrooms, or three-quarter bathrooms. calculatedbathnbr will be dropped due to the lack of clarity on the meaning of this column, the small number of null values, and the presence of data already captured by bathroomcnt.
    * **finishedsquarefeet12** - Contains a large number of null values. This feature is nearly identical to calculatedfinishedsquarefeet, which has fewer null values. 
    * **propertyzoningdesc** - Contains a large number of null values. The property zoning description has a large number of categorical values. One-hot encoding would result in an substantial increase in the number of features for any future modeling. These zoning descriptions can represent significantly different situations, therefore imputing these values is not recommended. 
    * **regionidcity** - Similar to propertyzoningdesc, these categorical values can represent significantly different characteristics and aren't simple to impute. Rather than losing the rows that have missing values for this feature, this column will be removed, as there are other similar features providing geographical information.
    * **roomcnt** - Over 50% of these are recorded as 0. It is nonsensical to have a property with 0 rooms, which means that these zeroes represent null values. This column exceeds our 50% threshold for null values. 
    * **unitcnt** - Data from this column was used to identify and eliminate properties that had a unit count higher than 1. At this point, the remaining data within this column is identical and will provide no insight during modeling. 
7. Records containing any missing value in the following columns were dropped:
    * **censustractandblock** 
    * **regionidzip**
8. Categorical variables not stored as object type were converted to objects.
9. Numeric codes in the fips column were converted to their relevant county name.
10. Outliers were defined as values exceeding six times the interquartile range. Values exceeding this threshold were squeezed (i.e. they were made equal to the threshold).
11. Remaining missing values were imputed as either the mean, median, or mode of each column. Generally, median was used for skewed distributions, mean was used for more symmetrically distributed data, and mode was used for categorical variables.
    * **buildingqualitytypeid** - Mean
    * **calculatedfinishedsquarefeet** - Median
    * **fullbathcnt** - Median
    * **heatingorsystemtypeid** - Mode
    * **heatingorsystemdesc** - Mode
    * **lotsizesquarefeet** - Median
    * **yearbuilt** - Mean
    * **structuretaxvaluedollarcnt** - Median
    * **taxvaluedollarcnt** - Median
    * **landtaxvaluedollarcnt** - Median
    * **taxamount** - Median

## How to Reproduce

### First clone this repo

### acquire.py 
* Must include `env.py` file in directory.
    * Contact [Codeup](https://codeup.com/contact/) to request access to the MySQL Server that the data is stored on.
    * `env.py` should include the following variables
        * `user` - should be your username
        * `password` - your password
        * `host` - the host address for the MySQL Server

### acquire.py
* The functions in prepare.py can be imported to another file. Each function is specific to the task developed during the data science pipeline of this project and may need to be altered to suit different purposes. 
### explore.ipynb
* The one_hot_encoding() function can be imported for more broad application, but the function will be unable to process a dataframe with dtypes other than numeric and/or object.
* The drop_object_variables() function is highly specific to this exploration and will need to be significantly modified for broader application.
* The visualize_regions() function is too specific to be used in applications outside of the specific use in this project.