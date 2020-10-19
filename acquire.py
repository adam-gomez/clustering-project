import env
import pandas as pd 
from os import path 
import math
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, QuantileTransformer, PowerTransformer, RobustScaler, MinMaxScaler

def get_connection(database):
    '''
    Database: string; name of database that the url is being created for
    '''
    return f'mysql+pymysql://{env.user}:{env.password}@{env.host}/{database}'

sql_query = '''
SELECT prop.*, logerror, transactiondate, airconditioningdesc, architecturalstyledesc, buildingclassdesc, heatingorsystemdesc, propertylandusedesc, storydesc, typeconstructiondesc
FROM properties_2017 as prop 
INNER JOIN (
	SELECT id, p.parcelid, logerror, transactiondate
	FROM predictions_2017 AS p
	INNER JOIN (
		SELECT parcelid,  MAX(transactiondate) AS max_date
		FROM predictions_2017 
		GROUP BY (parcelid)) AS sub
			ON p.parcelid = sub.parcelid
		WHERE p.transactiondate = sub.max_date
		) AS subq
	ON prop.parcelid = subq.parcelid
LEFT JOIN airconditioningtype
	ON prop.airconditioningtypeid = airconditioningtype.airconditioningtypeid
LEFT JOIN architecturalstyletype
	ON prop.architecturalstyletypeid = architecturalstyletype.architecturalstyletypeid
LEFT JOIN buildingclasstype 
	ON prop.buildingclasstypeid = buildingclasstype.buildingclasstypeid
LEFT JOIN heatingorsystemtype
	ON prop.heatingorsystemtypeid = heatingorsystemtype.heatingorsystemtypeid
LEFT JOIN propertylandusetype
	ON prop.propertylandusetypeid = propertylandusetype.propertylandusetypeid
LEFT JOIN storytype
	ON prop.storytypeid = storytype.storytypeid
LEFT JOIN typeconstructiontype
	ON prop.typeconstructiontypeid = typeconstructiontype.typeconstructiontypeid
WHERE latitude IS NOT NULL 
	AND longitude IS NOT NULL
	AND transactiondate BETWEEN '2017-01-01' AND '2017-12-31'
    AND prop.propertylandusetypeid IN (261, 262, 263, 264, 266, 268, 269, 273, 275, 276, 279)
'''

def acquire_cache_zillow():
    if not path.isfile('zillow.csv'):
        url = get_connection('zillow')
        zillow = pd.read_sql(sql_query, url)
        zillow.to_csv('zillow.csv')
    zillow = pd.read_csv('zillow.csv')
    zillow.drop(columns=['Unnamed: 0'], inplace=True)
    return zillow

def handle_missing_values(df, prop_required_column = .5, prop_required_row = .75):
    threshold = int(round(prop_required_column*len(df.index),0))
    df.dropna(axis=1, thresh=threshold, inplace=True)
    threshold = int(round(prop_required_row*len(df.columns),0))
    df.dropna(axis=0, thresh=threshold, inplace=True)
    return df

def map_county(row):
    if row['fips'] == 6037:
        return 'Los Angeles'
    elif row['fips'] == 6059:
        return 'Orange'
    elif row['fips'] == 6111:
        return 'Ventura'

def wrangle_zillow(df):
    df = df[df.bedroomcnt > 0]
    df = df[df.bathroomcnt > 0]
    df = df[~(df.unitcnt > 1)]
    df = df[df.calculatedfinishedsquarefeet > 0]
    df = df.loc[:,~df.columns.duplicated()]
    handle_missing_values(df)
    df.drop(columns=['id', 'parcelid', 'assessmentyear', 'calculatedbathnbr', 'finishedsquarefeet12', 'propertyzoningdesc', 'regionidcity', 'roomcnt', 'unitcnt'], inplace=True)
    df = df.dropna(subset=['censustractandblock'])
    df = df.dropna(subset=['regionidzip'])
    columns_to_change = ['fips', 'heatingorsystemtypeid', 'propertylandusetypeid', 'rawcensustractandblock', 'regionidcounty', 'regionidzip', 'censustractandblock']
    for column in columns_to_change:
    	df[column] = df[column].astype(object)
    df['fips'] = df.apply(lambda row: map_county(row), axis = 1)
    return df

def split_impute_zillow(df, pct=0.10):
	'''
	This divides the zillow dataframe into train, validate, and test sets.
	It then imputes missing values based on previously established best practice.
	
	Parameters - (df, pct=0.10)
    df = dataframe you wish to split
    pct = size of the test set, 1/2 of size of the validate set

    Returns three dataframes (train, validate, test)
	'''
	train_validate, test = train_test_split(df, test_size=pct, random_state = 123)
	train, validate = train_test_split(train_validate, test_size=pct*2, random_state = 123)

	train['buildingqualitytypeid'].fillna(value=train.buildingqualitytypeid.mean(), inplace=True)
	validate['buildingqualitytypeid'].fillna(value=train.buildingqualitytypeid.mean(), inplace=True)
	test['buildingqualitytypeid'].fillna(value=train.buildingqualitytypeid.mean(), inplace=True)

	train['calculatedfinishedsquarefeet'].fillna(value=train.calculatedfinishedsquarefeet.median(), inplace=True)
	validate['calculatedfinishedsquarefeet'].fillna(value=train.calculatedfinishedsquarefeet.median(), inplace=True)
	test['calculatedfinishedsquarefeet'].fillna(value=train.calculatedfinishedsquarefeet.median(), inplace=True)

	train['fullbathcnt'].fillna(value=train.fullbathcnt.median(), inplace=True)
	validate['fullbathcnt'].fillna(value=train.fullbathcnt.median(), inplace=True)
	test['fullbathcnt'].fillna(value=train.fullbathcnt.median(), inplace=True)

	train['heatingorsystemtypeid'].fillna(value=train.heatingorsystemtypeid.value_counts().index[0], inplace=True)
	validate['heatingorsystemtypeid'].fillna(value=train.heatingorsystemtypeid.value_counts().index[0], inplace=True)
	test['heatingorsystemtypeid'].fillna(value=train.heatingorsystemtypeid.value_counts().index[0], inplace=True)	

	train['heatingorsystemdesc'].fillna(value=train.heatingorsystemdesc.value_counts().index[0], inplace=True)
	validate['heatingorsystemdesc'].fillna(value=train.heatingorsystemdesc.value_counts().index[0], inplace=True)
	test['heatingorsystemdesc'].fillna(value=train.heatingorsystemdesc.value_counts().index[0], inplace=True)	

	train['lotsizesquarefeet'].fillna(value=train.lotsizesquarefeet.median(), inplace=True)
	validate['lotsizesquarefeet'].fillna(value=train.lotsizesquarefeet.median(), inplace=True)
	test['lotsizesquarefeet'].fillna(value=train.lotsizesquarefeet.median(), inplace=True)	

	train['yearbuilt'].fillna(value=train.yearbuilt.mean(), inplace=True)
	validate['yearbuilt'].fillna(value=train.yearbuilt.mean(), inplace=True)
	test['yearbuilt'].fillna(value=train.yearbuilt.mean(), inplace=True)	

	train['structuretaxvaluedollarcnt'].fillna(value=train.structuretaxvaluedollarcnt.median(), inplace=True)
	validate['structuretaxvaluedollarcnt'].fillna(value=train.structuretaxvaluedollarcnt.median(), inplace=True)
	test['structuretaxvaluedollarcnt'].fillna(value=train.structuretaxvaluedollarcnt.median(), inplace=True)	

	train['taxvaluedollarcnt'].fillna(value=train.taxvaluedollarcnt.median(), inplace=True)
	validate['taxvaluedollarcnt'].fillna(value=train.taxvaluedollarcnt.median(), inplace=True)
	test['taxvaluedollarcnt'].fillna(value=train.taxvaluedollarcnt.median(), inplace=True)
	
	train['landtaxvaluedollarcnt'].fillna(value=train.landtaxvaluedollarcnt.median(), inplace=True)
	validate['landtaxvaluedollarcnt'].fillna(value=train.landtaxvaluedollarcnt.median(), inplace=True)
	test['landtaxvaluedollarcnt'].fillna(value=train.landtaxvaluedollarcnt.median(), inplace=True)

	train['taxamount'].fillna(value=train.taxamount.median(), inplace=True)
	validate['taxamount'].fillna(value=train.taxamount.median(), inplace=True)
	test['taxamount'].fillna(value=train.taxamount.median(), inplace=True)
	
	return train, validate, test

def compress_outliers(df):
    '''
    Takes in a dataframe and identifies the interquartile range (IQR) of each numeric column.
    An upper threshold is defined for each column equal to the 75th percentile + 6 * IQR.
    A lower threshold is defined for each column equal to the 25th percentile - 6 * IQR.
    Any values above the upper threshold are set to be equal to the upper threshold.
    Any values below the lower threshold are set to be equal to the lower threshold.
    Returns the modified dataframe.
    '''
    columns = df.columns.to_list()
    for column in columns:
        if df[column].dtype == 'object':
            continue
        else:
            q1, q3 = df[column].quantile([.25, .75])
            IQR = q3 - q1
            upper_threshold = q3 + 6 * IQR
            lower_threshold = q1 - 6 * IQR 
            df[column] = df[column].apply(lambda x: upper_threshold if x > upper_threshold else x)
            df[column] = df[column].apply(lambda x: lower_threshold if x < lower_threshold else x)
    return df

def prepare_zillow():
    df = acquire_cache_zillow()
    df = wrangle_zillow(df)
    df = compress_outliers(df)
    train, validate, test = split_impute_zillow()
    return df


def standard_scaler(train, validate, test):
    '''
    Accepts three dataframes and applies a standard scaler to convert values in each dataframe
    based on the mean and standard deviation of each dataframe respectfully. 
    Columns containing object data types are dropped, as strings cannot be directly scaled.

    Parameters (train, validate, test) = three dataframes being scaled
    
    Returns (scaler, train_scaled, validate_scaled, test_scaled)
    '''
    # Remove columns with object data types from each dataframe
    train = train.select_dtypes(exclude=['object'])
    validate = validate.select_dtypes(exclude=['object'])
    test = test.select_dtypes(exclude=['object'])
    # Fit the scaler to the train dataframe
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True).fit(train)
    # Transform the scaler onto the train, validate, and test dataframes
    train_scaled = pd.DataFrame(scaler.transform(train), columns=train.columns.values).set_index([train.index.values])
    validate_scaled = pd.DataFrame(scaler.transform(validate), columns=validate.columns.values).set_index([validate.index.values])
    test_scaled = pd.DataFrame(scaler.transform(test), columns=test.columns.values).set_index([test.index.values])
    return scaler, train_scaled, validate_scaled, test_scaled

def scale_inverse(scaler, train_scaled, validate_scaled, test_scaled):
    '''
    Takes in three dataframes and reverts them back to their unscaled values

    Parameters (scaler, train_scaled, validate_scaled, test_scaled)
    scaler = the scaler you with to use to transform scaled values to unscaled values with. Presumably the scaler used to transform the values originally. 
    train_scaled, validate_scaled, test_scaled = the dataframes you wish to revert to unscaled values

    Returns train_unscaled, validated_unscaled, test_unscaled
    '''
    train_unscaled = pd.DataFrame(scaler.inverse_transform(train_scaled), columns=train_scaled.columns.values).set_index([train_scaled.index.values])
    validate_unscaled = pd.DataFrame(scaler.inverse_transform(validate_scaled), columns=validate_scaled.columns.values).set_index([validate_scaled.index.values])
    test_unscaled = pd.DataFrame(scaler.inverse_transform(test_scaled), columns=test_scaled.columns.values).set_index([test_scaled.index.values])
    return train_unscaled, validate_unscaled, test_unscaled


