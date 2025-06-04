# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 13:45:50 2021

@author: msagming
"""

def Preprocessing():
    
    import numpy as np
    import pandas as pd
    #=======================================================================================================
    # Read train data into a dataframe
    data = pd.read_table('orange_small_train.xls')
    output_labels = data.iloc[:,0:3] #Read output labels only
    input_data = data.iloc[:,3:233] #Read input data only
    
    row_hundred_percent_nan_index = [] #list to store rows with all entries as NaN
    column_seventy_percent_nan_index = [] #list to store columns with more than 70% entries as NaN
    
    #=======================================================================================================
    for i in range(3):
        if i==0:
            #Check the balance in the churn label
            churn_label = data.iloc[:,0]
            print(pd.Series(churn_label).value_counts())
        elif i==1:
            #Check the balance in the appentency label
            appetency_label = data.iloc[:,1]
            print(pd.Series(appetency_label).value_counts())
        else:
            #Check the balance in the upselling label
            upselling_label = data.iloc[:,2]
            print(pd.Series(upselling_label).value_counts())
    
    #=======================================================================================================
    #Find and delete columns containing more than 70% NaN values
    for i in range(len(input_data.columns)):
        column_data = input_data.iloc[:,i]
        nan_count = column_data.isna().sum()
        if nan_count > 35000:
           column_seventy_percent_nan_index.append(i) 
    input_data.drop(input_data.columns[column_seventy_percent_nan_index], axis = 1, inplace=True)
    
    #Find and delete all rows containing all entries as NaN
    for i in range(len(input_data.values)):
        row_data = input_data.iloc[i,:]
        nan_count = row_data.isna().sum()
        if nan_count == 50000:
           row_hundred_percent_nan_index.append(i) 
    input_data.drop(input_data.values[row_hundred_percent_nan_index], axis = 0, inplace=True)
    
    #=======================================================================================================
    #Feature Engineering
    #Compute number of NaN per row
    #Binary Feature to indicate the presence or absence of NaN values
    row_nan_count = []
    row_nan_binary = []
    present = 1
    absent = 0
    for i in range(len(input_data.values)):
        row_data = input_data.iloc[i,:]
        nan_count = row_data.isna().sum()
        if nan_count > 0:
            row_nan_count.append(nan_count) 
            row_nan_binary.append(present)
        else:
            row_nan_count.append(nan_count) 
            row_nan_binary.append(absent)
    #Add number of NaN per row and binary presence of NaN as new columns
    input_data['Number of NaN'] = row_nan_count
    input_data['Bool NaN'] = row_nan_binary
    
    #=======================================================================================================
    #Handling numerical features
    #We use SimpleImputer to handle missing values
    from sklearn.impute import SimpleImputer
    
    numerical_data = input_data.iloc[:,0:42] #Extract columns with numerical values 
    categorical_data = input_data.iloc[:,42:74] #Extract columns with categorical values 
    
    numerical = 1 #Choose how to replace NaN values
    
    if numerical==1:  
        #Replace NaN values with the mean of the column
        imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
        imputer.fit(numerical_data)
        numerical_data = imputer.transform(numerical_data)
    elif numerical==2:
        #Replace NaN values with the median of the column
        imputer = SimpleImputer(missing_values = np.nan, strategy = 'median')
        imputer.fit(numerical_data)
        numerical_data = imputer.transform(numerical_data)
    elif numerical==3:
        #Replace NaN values with the most frequent value of the column
        imputer = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')
        imputer.fit(numerical_data)
        numerical_data = imputer.transform(numerical_data)
    else:
        #Replace NaN values with a constant value of zero
        imputer = SimpleImputer(missing_values = np.nan, strategy = 'constant', fill_value=0)
        imputer.fit(numerical_data)
        numerical_data = imputer.transform(numerical_data)
        
    #=======================================================================================================
    #Handling categorical features
    categorical = 1 #Choose how to replace NaN values
    
    if categorical==1:
        #Replace NaN values with the most frequent value of the column
        imputer = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')
        imputer.fit(categorical_data)
        categorical_data = imputer.transform(categorical_data)
    else:
        #Replace NaN values with a constant value of zero
        imputer = SimpleImputer(missing_values = np.nan, strategy = 'constant', fill_value=0)
        imputer.fit(categorical_data)
        categorical_data = imputer.transform(categorical_data)
    #=======================================================================================================
    #Categorical data Encoding 
    category_encoding = 3
    import category_encoders as ce
    
    if category_encoding == 1:
        # (1) One Hot Encoder
        encoder= ce.OneHotEncoder(cols=None, return_df=True, use_cat_names=(False))
        categorical_data = encoder.fit_transform(categorical_data)
    elif category_encoding == 2:
        # (2) Dummy Encoding
        categorical_data = pd.get_dummies(data=categorical_data, drop_first=True)
    else: 
        # (3) Harshing Encoding
        encoder=ce.HashingEncoder(cols=None, n_components=32)
        categorical_data = encoder.fit_transform(categorical_data)
    
    #Convert Numpy Array to dataframe
    numerical_data = pd.DataFrame(numerical_data)
    #Join dataframes including the labels
    new_input_data = pd.concat([numerical_data,categorical_data], axis=1)
    new_input_data['Number of NaN'] = row_nan_count
    new_input_data['Bool NaN'] = row_nan_binary
    #new_input_data = pd.concat([new_input_data,output_labels], axis=1)
    
    #=======================================================================================================
    #Feature Selection
    #We select important features to improve the performance of the model
    X = new_input_data #Input data
    X.columns = X.columns.astype(str)

    Y1 = output_labels.iloc[:,0] #Churn
    Y2 = output_labels.iloc[:,1] #Appetency
    Y3 = output_labels.iloc[:,2] #Up-Selling

    from numpy import arange
    from sklearn.model_selection import GridSearchCV, RepeatedKFold
    from sklearn.linear_model import Lasso
    from sklearn.preprocessing import StandardScaler
    
    # scale X
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # define model
    model = Lasso(max_iter=10000)
    # define model evaluation method
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

    # define grid
    grid = dict()
    grid['alpha'] = arange(0.01, 1, 0.01)  # Không để alpha=0

    # define and perform search
    search_1 = GridSearchCV(model, grid, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
    results_1 = search_1.fit(X, Y1)

    search_2 = GridSearchCV(model, grid, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
    results_2 = search_2.fit(X, Y2)

    search_3 = GridSearchCV(model, grid, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
    results_3 = search_3.fit(X, Y3)

    # summary
    print('------------------------------------')
    print('Churn')
    print('MAE: %.3f' % results_1.best_score_)
    best_param_1 = results_1.best_params_['alpha']
    print('Alpha: %s' % best_param_1)
    print('------------------------------------')
    print('Appetency')
    print('MAE: %.3f' % results_2.best_score_)
    best_param_2 = results_2.best_params_['alpha']
    print('Alpha: %s' % best_param_2)
    print('------------------------------------')
    print('Up-Selling')
    print('MAE: %.3f' % results_3.best_score_)
    best_param_3 = results_3.best_params_['alpha']
    print('Alpha: %s' % best_param_3)
    print('------------------------------------')   
    
    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    #Churn split
    X1_train, X1_test, Y1_train, Y1_test = train_test_split(X, Y1, test_size = 0.2, random_state = 0)
    #Appentency slpit
    X2_train, X2_test, Y2_train, Y2_test = train_test_split(X, Y2, test_size = 0.2, random_state = 0)
    #Up-Selling slpit
    X3_train, X3_test, Y3_train, Y3_test = train_test_split(X, Y3, test_size = 0.2, random_state = 0)

    #Using the best alpha values and Lasso Regression to select the best parameters
    #Churn best parameters
    best_param_1=0.02
    regression_1 = Lasso(alpha=best_param_1)
    regression_1 = regression_1.fit(X1_train, Y1_train)
    coefficients_1 = pd.DataFrame(regression_1.coef_,columns=['coef'])
    coefficients_1['variables'] = X1_train.columns
    selected_features_1 = list(coefficients_1[abs(coefficients_1['coef'])>0].variables)
    X1_selected = X[selected_features_1]
    churn_input_data = pd.concat([X1_selected,Y1], axis=1)
    
    #Appetency best parameters
    regression_2 = Lasso(alpha=best_param_2)
    regression_2 = regression_2.fit(X2_train, Y2_train)
    coefficients_2 = pd.DataFrame(regression_2.coef_,columns=['coef'])
    coefficients_2['variables'] = X2_train.columns
    selected_features_2 = list(coefficients_2[abs(coefficients_2['coef'])>0].variables)
    X2_selected = X[selected_features_2]
    appetency_input_data = pd.concat([X2_selected,Y2], axis=1)

    
    #Up-Selling best parameters
    regression_3 = Lasso(alpha=best_param_3)
    regression_3 = regression_3.fit(X3_train, Y3_train)
    coefficients_3 = pd.DataFrame(regression_3.coef_,columns=['coef'])
    coefficients_3['variables'] = X3_train.columns
    selected_features_3 = list(coefficients_3[abs(coefficients_3['coef'])>0].variables)
    X3_selected = X[selected_features_3]
    upselling_input_data = pd.concat([X3_selected,Y3], axis=1)
    
    #==================================================================================================

    #write churn hyperparamets to file to file
    filename = 'Churn_tuned_dataset.csv'
    churn_input_data.to_csv(filename, index=False, header=False)
    
    #write appentency hyperparamets to file to file
    filename = 'Appetency_tuned_dataset.csv'
    appetency_input_data.to_csv(filename, index=False, header=False)
    
    #write up-selling hyperparamets to file to file
    filename = 'Upselling_tuned_dataset.csv'
    upselling_input_data.to_csv(filename, index=False, header=False)
    
#==================================================================================================
 
    
if __name__ == '__main__':
    Preprocessing()  