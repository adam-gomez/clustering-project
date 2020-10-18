# Predicting Log Error from Zillow's Zestimate

This project is an extension of a previous project. View the github of the previous project here: [**Predicting Tax Assessed Home Values**](https://github.com/Estimating-Home-Value/regression-project)

## Description
> For this project you will continue working with the zillow dataset. Continue to use the 2017 properties and predictions data for single unit / single family homes.

> In addition to continuing work on your previous project, you should incorporate clustering methodologies on this project.

> Your audience for this project is a data science team. The presentation will consist of a notebook demo of the discoveries you made and work you have done related to uncovering what the drivers of the error in the zestimate is.

## Goals
1. Identify significant drivers of log error for single unit properties in 2017
2. Lorem ipsum dolor sit amet, consectetur adipiscing elit. 
3. Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas. 
4. Aenean viverra accumsan massa, vitae tincidunt risus laoreet sed. Donec fermentum, mauris quis porttitor mollis, ante magna hendrerit lorem, in blandit risus turpis id nisi.

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

## Data Dictionary
| Column | Description |
| --- | ---|
| id | Autoincremented unique index id for each property |
| bathroomcnt | Number of Bathrooms; Includes halfbaths as 0.5 |
| bedroomcnt | Number of Bedrooms |
| calculatedbathnbr | Precise meaning unknown, but appears to be redundant with bathroomcnt and bedroomcnt |
| calculatedfinishedsquarefeet | Total square feet of home; doesn't include property square feet |
| finishedsquarefeet12| Unknown, but appears to be redundant with calculatedfinishedsquarefeet | 
| fips | Federal Information Processing System codes used to identify unique geographical areas | 
| fullbathcnt | Number of full bathrooms |
| latitude | The latitude of the property
| longitude | The longitude of the property |
| lotsizesquarefeet| The size of the total property lot |
| propertycountylandusecode | Unknown, but represents categorical government code |
| propertylandusetypeid |  Categorical variable describing the general type of property |
| rawcensustractandblock | Government id for each property linked to geographic location |
| regionidcity | Categorical variable identifying geographic location |
| regionidcounty | Categorical variable identifying geographic location |
| roomcnt | Number of rooms |
| yearbuilt | The year the house was built |
| structuretaxvaluedollarcnt | The tax assessed value of only the property structure in USD | 
| assessmentyear | Year that the tax value was assessed |
| landtaxvaluedollarcnt | The tax assessed value of only the land lot for the property |
| taxamount | The amount paid in taxes by the landowner in USD |
| taxvaluedollarcnt | The tax accessed value of the property in USD |
| censustractandblock | Redundant with rawcensustractandblock |
| logerror | Unknown |
| transactiondate | Four digit year, two digit month, two digit date | 
| taxrate | Rounded derived value by dividing the taxamount by the taxvaluedollarcnt and multiplying by 100 |
| County | County the property is located in | 

Additional columns were present in the zillow database but had greater than 20% null values and were dropped during initial consideration. 

## Data Validation
The following considerations were taken with the data:
1. Initial SQL query produced 20,364 records that met the following requirements:
    * Transaction Date between 2017-01-01 and 2017-12-31
    * Property classified as one of the following Types:
        * Single Family Residential
        * Rural Residence
        * Mobile Home
        * Townhouse
        * Condominium
        * Row House
        * Bungalow
        * Manufactured, Modular, Prefabricated Homes
        * Patio Home
        * Inferred Single Family Residence
2. Records containing 0 bathrooms, 0 bedrooms, or null square footage were dropped
3. Duplicate records were dropped. These entries may represent "back-to-back" closings on the same day between three parties.

## Managing Outliers
Outliers were identified in the following features. Due to the significant influence of outliers on clustering techniques, the following considerations were made:
- Feature 1: Consideration
- Feature 2: Consideration
- Feature 3: Consideration
- Feature 4: Consideration

## Key Findings and Takeways:
1. Aenean faucibus purus nec felis vehicula, vel varius orci tempus. 
2. Liquam feugiat ipsum non enim efficitur ornare. Nunc id sapien interdum tortor eleifend volutpat sed ut risus.
3. Maecenas pulvinar nisl lacinia neque pretium aliquam. 
4. Phasellus faucibus, justo eu vehicula semper, ipsum urna tincidunt magna, quis eleifend sem orci sed arcu.