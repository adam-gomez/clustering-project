import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.preprocessing import StandardScaler, QuantileTransformer, PowerTransformer, RobustScaler, MinMaxScaler

def corr_plot(df):
    """
    Takes in a dataframe and returns a correlation plot of all of the numeric variables
    """
    all_features = df.columns.to_list()
    plotted_features = []
    for feature in all_features:
        if df[feature].dtype != 'object':
            plotted_features.append(feature)
    corr = df[plotted_features].corr()
    plt.rc('font',size=11)
    plt.rc('figure', figsize=(13,7))
    sns.heatmap(corr, cmap='Blues', annot=True)

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

def plot_pairs(df):
    g = sns.PairGrid(df)
# we can specify any two functions we want for visualization
    g.map_diag(plt.hist) # single variable
    g.map_offdiag(sns.regplot, scatter_kws={"color": "steelblue"}, line_kws={"color": "cyan"}) # interaction of two variables

def one_hot_encoding(df, features):
    '''
    Takes in a dataframe (df) and a list of categorical (object type) features (features) to encode as numeric dummy variables, then drops the
    original listed feature columns from the dataframe.
    
    Returns the dataframe
    '''
    for feature in features:
        df[feature] = df[feature].astype(object)
    obj_df = df[features]
    dummy_df = pd.get_dummies(obj_df, dummy_na=False, drop_first=True)
    df = pd.concat([df, dummy_df], axis=1)
    df.drop(columns=features, inplace=True)
    return df