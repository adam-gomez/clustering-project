import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.preprocessing import StandardScaler, QuantileTransformer, PowerTransformer, RobustScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, RFE, f_regression 
from sklearn.linear_model import LinearRegression

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

def select_kbest(X_train_scaled, y_train, k):
    '''
    Takes in the predictors (X_train_scaled), the target (y_train), 
    and the number of features to select (k) 
    and returns the names of the top k selected features based on the SelectKBest class
    '''
    f_selector = SelectKBest(f_regression, k)
    f_selector = f_selector.fit(X_train_scaled, y_train)
    X_train_reduced = f_selector.transform(X_train_scaled)
    f_support = f_selector.get_support()
    f_feature = X_train_scaled.iloc[:,f_support].columns.tolist()
    return f_feature

def rfe(X_train_scaled, y_train, k):
    '''
    Takes in the predictor (X_train_scaled), the target (y_train), 
    and the number of features to select (k).
    Returns the top k features based on the RFE class.
    '''
    lm = LinearRegression()
    rfe = RFE(lm, k)
    # Transforming data using RFE
    X_rfe = rfe.fit_transform(X_train_scaled, y_train)
    #Fitting the data to model
    lm.fit(X_rfe,y_train)
    mask = rfe.support_
    rfe_features = X_train_scaled.loc[:,mask].columns.tolist()
    return rfe_features

def drop_object_variables(train, validate, test):
    '''
    Drops the object type variables from the train, validate, and test zillow dataframe.
    This function is not universal, it is specific to the dataframe it was designed for.
    '''
    train.drop(columns=['heatingorsystemtypeid', 'cluster2', 'cluster3'], inplace=True)
    validate.drop(columns=['heatingorsystemtypeid'], inplace=True)
    test.drop(columns=['heatingorsystemtypeid'], inplace=True)
    train = train.select_dtypes(exclude=['object'])
    validate = validate.select_dtypes(exclude=['object'])
    test = test.select_dtypes(exclude=['object'])
    return train, validate, test

def create_scaled_x_y(train, validate, test, target):
    '''
    Accepts three dataframes (train, validate, test) and a target variable. 
    Separates the target variable from the dataframes, scales train, validate, and test
    and returns all 6 resulting dataframes.
    '''
    y_train = train[target]
    X_train = train.drop(columns=[target])
    y_validate = validate[target]
    X_validate = validate.drop(columns=[target])
    y_test = test[target]
    X_test = test.drop(columns=[target])
    scaler, X_train_scaled, X_validate_scaled, X_test_scaled = standard_scaler(X_train, X_validate, X_test)
    return X_train_scaled, y_train, X_validate_scaled, y_validate, X_test_scaled, y_test

def visualize_regions(kmeans, X, df, cluster_name):
    '''
    This functions accepts a developed kmeans model with the dataframe used to develop the model (X)
    And returns a visualization of the clusters within that dataframe based on the kmeans model
    '''
    centroids = pd.DataFrame(kmeans.cluster_centers_, columns=X.columns)
    plt.figure(figsize=(14,9))
    for cluster, subset in df.groupby(cluster_name):
        plt.scatter(subset.longitude, subset.latitude, label='region ' + str(cluster), alpha = .6)
    plt.legend()
    plt.xlabel('latitude')
    plt.ylabel('longitude')
    plt.title('Visualizing Regions')
    plt.show()

def baseline_rmse(y_train):
    '''
    Computes the root mean squared error of a baseline prediction where the baseline model
    is the mean of the array.
    '''
    baseline_residual = y_train - y_train.mean()
    baseline_sse = (baseline_residual**2).sum()
    baseline_mse = baseline_sse/y_train.size
    rmse = math.sqrt(baseline_mse)
    return rmse

def plot_residuals(actual, predicted, feature):
    """
    Returns the scatterplot of actural y in horizontal axis and residuals in vertical axis
    Parameters: actural y(df.se), predicted y(df.se), feature(str)
    Prerequisite: call function evaluate_slr
    """
    plt.figure(figsize=(20,10))
    residuals = actual - predicted
    plt.hlines(0, actual.min(), actual.max(), ls=':')
    plt.scatter(actual, residuals)
    plt.ylabel('residual ($y - \hat{y}$)')
    plt.xlabel('actual value ($y$)')
    plt.title(f'Actual vs Residual on {feature}')
    return plt.gca()

def plot_residuals_percentage(actual, predicted, feature):
    """
    Returns the scatterplot of actural y in horizontal axis and residuals in vertical axis
    Parameters: actural y(df.se), predicted y(df.se), feature(str)
    Prerequisite: call function evaluate_slr
    """
    residuals = actual - predicted
    residuals_percentage = residuals/actual
    plt.hlines(0, actual.min(), actual.max(), ls=':')
    plt.scatter(actual, residuals_percentage)
    plt.ylabel('residual ($y - \hat{y}$)%')
    plt.xlabel('actual value ($y$)')
    plt.title(f'Actual vs Residual% on {feature}')
    return plt.gca()

def validate_rmse(y_train, y_validate):
    '''
    Computes the root mean squared error of the baseline model (the mean of y_train) 
    when compared to the y_validate targets
    '''
    baseline_validate_residual = y_validate - y_train.mean()
    baseline_validate_sse = (baseline_validate_residual**2).sum()
    baseline_validate_mse = baseline_validate_sse/y_validate.size
    v_rmse = math.sqrt(baseline_validate_mse)
    return v_rmse

def test_rmse(y_train, y_test):
    '''
    Computes the root mean squared error of the baseline model (the mean of y_train)
    when compared to the y_test targets
    '''
    baseline_test_residual = y_test - y_train.mean()
    baseline_test_sse = (baseline_test_residual**2).sum()
    baseline_test_mse = baseline_test_sse/y_test.size
    t_rmse = math.sqrt(baseline_test_mse)
    return t_rmse 

