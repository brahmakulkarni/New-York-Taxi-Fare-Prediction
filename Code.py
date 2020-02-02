#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from xgboost import XGBRegressor, plot_importance
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import pickle
from collections import OrderedDict
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.datasets import make_regression
import warnings
warnings.filterwarnings("ignore")


# # Functions
# 

# ## Marks Invalid values in data in an 'Invalid' column

# In[4]:


coord_list = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'fare_amount']
def mark_invalid(chunk):
    length = len(chunk)
    invalid = [0]*length
    for c in coord_list:
        for i in chunk.index:
            if(c == "pickup_longitude" or c == "dropoff_longitude"):
                if(chunk[c][i].astype(float) > -73.699215 or chunk[c][i].astype(float) < -74.257159):
                    invalid[i%length] = 1
            elif (c == "pickup_latitude" or c == "dropoff_latitude"):
                if(chunk[c][i].astype(float) > 40.915568 or chunk[c][i].astype(float) < 40.495992):
                    invalid[i%length] = 1
            elif(c == "fare_amount"):
                if(chunk[c][i] >= 200 or chunk[c][i] <= 0):
                    invalid[i%length] = 1
    return invalid

coord_list_test = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']
def mark_invalid_test(chunk):
    length = len(chunk)
    invalid_test = [0]*length
    for c in coord_list_test:
        for i in chunk.index:
            if(c == "pickup_longitude" or c == "dropoff_longitude"):
                if(chunk[c][i].astype(float) > -73.699215 or chunk[c][i].astype(float) < -74.257159):
                    invalid_test[i%length] = 1
            elif (c == "pickup_latitude" or c == "dropoff_latitude"):
                if(chunk[c][i].astype(float) > 40.915568 or chunk[c][i].astype(float) < 40.495992):
                    invalid_test[i%length] = 1
    return invalid_test


# ## Marks outliers also in the 'invalid' column

# In[5]:


def mark_outlier(chunk, data_1):
    outliers_indices=[]
    threshold = 3
    mean_1 = np.mean(data_1)
    std_1 = np.std(data_1)
    
    for i in chunk.index:
        z_score= (data_1[i] - mean_1)/std_1 
        if np.abs(z_score) > threshold:
            outliers_indices.append(i)
    for i in outliers_indices:
        chunk['invalid'][i] = 1
    return chunk


# ## Splitting 'pickup_datetime' column into various relevant columns

# In[6]:


def split_datetime(chunk):
    hours = []
    mins = []
    secs = []
    years = []
    months = []
    days = []
    length = len(chunk['pickup_longitude'])
    
    for i in chunk.index:
        years.append(int(chunk['pickup_datetime'][i][0:4]))
        months.append(int(chunk['pickup_datetime'][i][5:7]) - 1) # 1 is subtracted to aid in days from jan 1st calculations
        days.append(int(chunk['pickup_datetime'][i][8:10]))
        hours.append(int(chunk['pickup_datetime'][i][11:13]))
        mins.append(int(chunk['pickup_datetime'][i][14:16]))
        secs.append(int(chunk['pickup_datetime'][i][17:19]))

    chunk['years'] = years
    chunk['months'] = months
    chunk['days'] = days
    chunk['hours'] = hours
    chunk['mins'] = mins
    chunk['secs'] = secs
    
    return chunk


# ## Manipulates split columns

# In[7]:


def modify_datetime(chunk):
    chunk['secs_past_midnight'] = (chunk['hours']*3600) + (chunk['mins']*60) + (chunk['secs'])
    chunk['sin_spm'] = np.sin(2*np.pi*(chunk['secs_past_midnight']/86400))
    chunk['cos_spm'] = np.cos(2*np.pi*(chunk['secs_past_midnight']/86400))
    chunk['days_past_jan1'] = (chunk['months']*30) + (chunk['days'])
    chunk['sin_dpj'] = np.sin(2*np.pi*(chunk['days_past_jan1']/365))
    chunk['cos_dpj'] = np.cos(2*np.pi*(chunk['days_past_jan1']/365))
    return chunk


# ## Splits train data into feature set and target

# In[8]:


def split_data(chunk):
    y = chunk['fare_amount']
    X = pd.DataFrame(chunk)
    X = X.drop(['fare_amount','key','pickup_datetime', 'years', 'months', 'days', 'mins', 'secs', 'secs_past_midnight', 'days_past_jan1'], axis = 1)
    X = StandardScaler().fit_transform(X)
    return (X, y)


# ## Performs test-train split, fits models and computes RMSE

# In[9]:


def fit_model_rmse(X, y, model = LinearRegression()):
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    X_train = X_train.values
    X_test = X_test.values
#     parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower
#               'objective':['reg:linear'],
#               'learning_rate': [.03, 0.05, .07], #so called `eta` value
#               'max_depth': [5, 6, 7],
#               'min_child_weight': [4],
#               'silent': [1],
#               'subsample': [0.7],
#               'colsample_bytree': [0.7],
#               'n_estimators': [500]}
#     xgb_grid = GridSearchCV(model,
#                         parameters,
#                         cv = 2,
#                         n_jobs = 5,
#                         verbose=True)
#     xgb_grid.fit(X_train,y_train)
#     linreg = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    return (model, rmse)

# X1_train, X1_test, y1_train, y1_test = train_test_split(features, target, random_state=1)


# ## Saves model to pickle file

# In[10]:


def save_model(model, filename):
    pickle.dump(model, open(filename, 'wb'))


# ## Loads model from pickle file

# In[11]:


def load_model(filename): 
    model = pickle.load(open(filename, 'rb'))
    return model


# ## Returns features required by model from test data

# In[12]:


def req_columns_test(chunk):
    temp = pd.DataFrame(chunk)
    temp = temp.drop(['key','pickup_datetime', 'years', 'months', 'days', 'mins', 'secs', 'secs_past_midnight', 'days_past_jan1'], axis = 1)
    return temp


# In[13]:


def generate_dist(chunk):
    chunk['manhattan_dist'] = abs(chunk['pickup_latitude']-chunk['dropoff_latitude']) + abs(chunk['pickup_longitude']-chunk['dropoff_longitude'])
#     chunk['dist_to_jfk'] = abs(chunk['pickup_latitude']-40.6413) + abs(chunk['pickup_longitude']-73.7781)
    return chunk


# In[13]:


model = load_model('model_xgb_final1.sav')
OrderedDict(sorted(model.get_booster().get_fscore().items(), key=lambda t: t[1], reverse=True))


# # Reading chunks, running all functions and generating model from train data

# In[21]:


df_chunk = pd.read_csv('train.csv', chunksize = 500000, low_memory = False)
chunks = []
count = 0
for chunk in df_chunk:
    chunk = pd.DataFrame(chunk)
    if (count == 0):
#         model = XGBRegressor(
#             nthread = 5
#             objective = 'reg:linear',
#             learning_rate = 0.03,
#             max_depth = 5,
#             min_child_weight = 4,
#             n_estimators = 750
#         )
        model = XGBRegressor()
    else:
        model = load_model("model_xgb_final.sav")
    
    chunk['invalid'] = mark_invalid(chunk)
    chunk.dropna(inplace = True)
    chunk = split_datetime(chunk)
    chunk = modify_datetime(chunk)
    chunk = mark_outlier(chunk, chunk['fare_amount'])
    chunk = generate_dist(chunk)
    (X1, y1) = split_data(chunk)
    X1 = pd.DataFrame(X1)
    model, rmse = fit_model_rmse(X1, y1, model)
    save_model(model, "model_xgb_final.sav")
    print("RMSE: ", rmse)
    
    if (count == 0):
        chunks.append(pd.DataFrame(chunk))
    count = count + 1
    if(count == 6):
        break


# # Reading chunks, running all functions and generating model from test data

# In[27]:


count = 0
test_chunk = pd.read_csv('test.csv', chunksize = 1000000, low_memory = False)

test_chunks = []
final_dfs = []
for chunk in test_chunk:
    chunk = pd.DataFrame(chunk)
    if(count == 0):
        model = load_model("model_xgb_jfk.sav")
    chunk = chunk.interpolate(method='linear', limit_direction = 'forward')
    chunk['invalid'] = mark_invalid_test(chunk)
    chunk = split_datetime(chunk)
    chunk = modify_datetime(chunk)
    chunk = generate_dist(chunk)
    if(count == 0):
        test_chunks.append(pd.DataFrame(chunk))
    req_df = req_columns_test(chunk)
    req_df = req_df.values
    y_pred = model.predict(req_df)
    df = pd.DataFrame({'key':chunk['key'], 'fare_amount': y_pred})
    final_dfs.append(df)
    
    count += 1
    print("Finished iteration: ", count)


# # Concatenating key and final predictions to make submission csv

# In[28]:


output = pd.concat(final_dfs)
output_status = output.to_csv("output_xgb_jfk.csv", index = False)


# In[16]:


test_chunks[0].head()


# # Visualization plots 

# In[19]:


fig, ax = plt.subplots()
print(chunks[0]["pickup_longitude"].plot.hist(ax = ax, title="pickup longitude",bottom=1, bins=25))
ax.set_yscale('log')


# In[20]:


fig, ax = plt.subplots()
print(chunks[0]["pickup_latitude"].plot.hist(ax = ax, title="pickup latitude",bottom=1, bins=25))
ax.set_yscale('log')


# In[21]:


fig, ax = plt.subplots()
print(chunks[0]["dropoff_longitude"].plot.hist(ax = ax, title="dropoff longitude",bottom=1, bins=25))
ax.set_yscale('log')


# In[22]:


fig, ax = plt.subplots()
print(chunks[0]["dropoff_latitude"].plot.hist(ax = ax, title="dropoff latitude",bottom=1, bins=25))
ax.set_yscale('log')


# In[23]:


fig, ax = plt.subplots()
print(chunks[0]["passenger_count"].plot.hist(ax = ax, title="passenger count",bottom=1, bins=25))
ax.set_yscale('log')


# In[24]:


fig, (ax1, ax2) = plt.subplots(1, 2, sharex = True, sharey = True)
ax1.scatter(chunks[0]["pickup_latitude"],chunks[0]["fare_amount"])
ax2.scatter(chunks[0]["pickup_longitude"],chunks[0]["fare_amount"])
plt.show()


# In[25]:


plt.scatter(chunks[0]["dropoff_latitude"],chunks[0]["fare_amount"])
plt.xlabel("dropoff_latitude")
plt.ylabel("fare_amount")
plt.show()


# In[26]:


plt.scatter(chunks[0]["dropoff_longitude"],chunks[0]["fare_amount"])
plt.xlabel("dropoff_longitude")
plt.ylabel("fare_amount")
plt.show()


# In[27]:


plt.scatter(chunks[0]["passenger_count"],chunks[0]["fare_amount"])
plt.xlabel("passenger_count")
plt.ylabel("fare_amount")
plt.show()


# In[26]:


temp_chunk = split_datetime(chunks[0])
temp_chunk = modify_datetime(temp_chunk)
temp_chunk['manhattan_dist'] = abs(chunk['pickup_latitude']-chunk['dropoff_latitude']) + abs(chunk['pickup_longitude']-chunk['dropoff_longitude'])

plt.scatter(temp_chunk["years"],temp_chunk["fare_amount"])
plt.xlabel("year")
plt.ylabel("fare_amount")
plt.show()
plt.savefig("year_scatter.png")


# # Detecting Outliers

# In[28]:


plt.boxplot(chunks[0].fare_amount)

