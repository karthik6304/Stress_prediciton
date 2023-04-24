#importing the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import matplotlib.dates as mdates
from scipy import signal

#please change the value to access subject details. eg: '02' , '08', '20'
num ='24'
time_logs = pd.read_excel('Time_logs.xlsx')
# /Users/karthikningala/CE888-DATA SCIENCE AND DECISION MAKING/Time_logs.xlsx
time_logs.head()
#dividing the time logs into intervals
s=time_logs.loc[time_logs['S. ID.']==f'S{num}']
stroop_test = s['Stroop Test']
relax1 = s['Relax']
interview=s['Interview']
relax2=s['Relax.1']
hyperventilation = s['Hyperventilation']
relax3 = s['Relax.2']

list_of_test = [stroop_test,relax1,interview,relax2,hyperventilation,relax3]
time_logs = []
stroop_test.astype(str).values[0]
for i in list_of_test:
    a = i.astype(str)
    a= a.values[0]
    print(a)
    time_logs.append(a)
time_logs = time_logs[:6]
# READING THE SUBJECTS HEART RATE VAUES
hrate = pd.read_csv(f'https://raw.githubusercontent.com/italha-d/Stress-Predict-Dataset/main/Raw_data/S{num}/HR.csv',header=None)
hrate.head()
# DATA ANALYSIS
hrate[2:].info()
#summary of heart rate excluding the first two rows
hrate[2:].describe()
# DATA PROCESSING
#initial time_stamp
hr_initital_time = int(hrate[0][0])
#creating a new pandas dataframe
hr_combined_data = pd.DataFrame()
#adding the heart_rate values to the dataframe
hr_combined_data['heart_rate'] = hrate[2:]
#creating the time_stamp for the rest of the values
hr_combined_data['utc_time']= range(hr_initital_time,hr_initital_time+len(hr_combined_data['heart_rate']))
#resetting the index to start with 0
hr_combined_data= hr_combined_data.reset_index()
hr_combined_data=hr_combined_data.drop(columns='index')
#converting the UTC time to 24 hour format time
hr_time_stp = []
for i in  hr_combined_data['utc_time']:
    c = str(datetime.datetime.fromtimestamp(i).time())
    d = datetime.datetime.strptime(c, "%H:%M:%S")
    a =d.strftime("%I:%M:%S" )
    hr_time_stp.append(a)
    # hr_time_stp.append(f'{c.hour}:{c.minute}:{c.second}')
hr_combined_data['time'] = hr_time_stp
#combined data
hr_combined_data
# PLOTTING THE DATA
#plotting the heart rate 
fig, a_x = plt.subplots(figsize=(40, 10))
#paramaters 
y=hr_combined_data['heart_rate']
x=hr_combined_data['time']

#marking time intervals
time = mdates.MinuteLocator(interval=250000)
a_x.xaxis.set_major_locator(time) 
#plotting the graph
a_x.plot(x.astype(str), y)
# a_x.plot(rolling_mean)
# a_x.plot(rolling_mean2)
plt.xlabel('time')
plt.ylabel('heart rate')
for t in time_logs:
    plt.axvline(x = t, color = 'b', label = 'axvline - full height')
plt.axhline(y = y.mean(), color = 'b', label = 'axvline - full height')

#the vertical lines are the sessions and and the horizontal line respresents the mean value of the data.
# READING THE SUBJECTS EDA VALUES
#reading the eda data
eda = pd.read_csv(f'https://raw.githubusercontent.com/italha-d/Stress-Predict-Dataset/main/Raw_data/S{num}/EDA.csv',header=None)
eda.head()
eda_data = eda[2:]
# DATA ANALYSIS
eda_data.info()
eda_data.describe()
# DATA PROCESSING
#filtering every 4th element as frequency is 4hz
count = 0
eda_4 = []
for i in eda_data[0]:
    if count % 4 == 0:
        eda_4.append(i)
    count +=1
#initital time stamp
eda_initital_time = int(eda[0][0])
#creating a pandas datafreame
eda_combined_data = pd.DataFrame()
#combining the filtered data to the dataframe
eda_combined_data['eda'] = eda_4
#creating time stamps for the rest of the data
eda_combined_data['utc_time']= range(eda_initital_time,eda_initital_time+len(eda_4))
eda_combined_data= eda_combined_data.reset_index()
eda_combined_data=eda_combined_data.drop(columns='index')
#converting UTC to 24 hr format
eda_time_stp = []
for i in  eda_combined_data['utc_time']:
    c = str(datetime.datetime.fromtimestamp(i).time())
    d = datetime.datetime.strptime(c, "%H:%M:%S")
    a =d.strftime("%I:%M:%S" )
    eda_time_stp.append(a)
    # eda_time_stp.append(f'{c.hour}:{c.minute}:{c.second}')
#combining the converted time to the dataframe
eda_combined_data['time'] = eda_time_stp
#combined data
eda_combined_data
# eda_combined_data= find_index(eda_combined_data)
#plotting the eda values
fig, a_x = plt.subplots(figsize=(40, 6))
#parameters
y=eda_combined_data['eda']
x=eda_combined_data['time']

#marking time intervals
loc = mdates.MinuteLocator(interval=250000)
a_x.xaxis.set_major_locator(loc) # Locator for major axis only.
#plotting the graph
a_x.plot(x.astype(str), y)
plt.xlabel('time')
plt.ylabel('eda')
for t in time_logs:
    plt.axvline(x = t, color = 'b', label = 'axvline - full height')
plt.axhline(y = y.mean(), color = 'b', label = 'axvline - full height')
#the vertical lines are the sessions and and the horizontal line respresents the mean value of the data.
temp = pd.read_csv(f'https://raw.githubusercontent.com/italha-d/Stress-Predict-Dataset/main/Raw_data/S{num}/TEMP.csv',header=None)
temp.head()
temp_data = temp[2:]
temp_data.head()
#filtering every 4th element as frequency is 4hz
count = 0
temp_4 = []
for i in temp_data[0]:
    if count % 4 == 0:
        temp_4.append(i)
    count +=1
#initital time stamp
temp_initital_time = int(temp[0][0])
#creating a pandas datafreame
temp_combined_data = pd.DataFrame()
#combining the filtered data to the dataframe
temp_combined_data['temp'] = temp_4
#creating time stamps for the rest of the data
temp_combined_data['utc_time']= range(temp_initital_time,temp_initital_time+len(temp_4))
temp_combined_data= temp_combined_data.reset_index()
temp_combined_data=temp_combined_data.drop(columns='index')
#converting UTC to 24 hr format
temp_time_stp = []
for i in  temp_combined_data['utc_time']:
    c = str(datetime.datetime.fromtimestamp(i).time())
    d = datetime.datetime.strptime(c, "%H:%M:%S")
    a =d.strftime("%I:%M:%S" )
    temp_time_stp.append(a)
    # eda_time_stp.append(f'{c.hour}:{c.minute}:{c.second}')
#combining the converted time to the dataframe
temp_combined_data['time'] = temp_time_stp
temp_combined_data
# temp_combined_data= find_index(temp_combined_data)
#plotting the eda values
fig, a_x = plt.subplots(figsize=(40, 6))
#parameters
y=temp_combined_data['temp']
x=temp_combined_data['time']
# rolling_mean_eda = y.rolling(window=200).mean()

#marking time intervals
loc = mdates.MinuteLocator(interval=250000)
a_x.xaxis.set_major_locator(loc) # Locator for major axis only.
#plotting the graph
a_x.plot(x.astype(str), y)
plt.xlabel('time')
plt.ylabel('temp')
# plt.plot(rolling_mean_eda)
for t in time_logs:
    plt.axvline(x = t, color = 'b', label = 'axvline - full height')
plt.axhline(y = y.mean(), color = 'b', label = 'axvline - full height')
#the vertical lines are the sessions and and the horizontal line respresents the mean value of the data.
# READING TAG VALUES
tags = pd.read_csv(f'https://raw.githubusercontent.com/italha-d/Stress-Predict-Dataset/main/Raw_data/S{num}/tags_S{num}.csv',header=None)
tags.head()
# mergind the heart rate, EDA and temperature values on time
final_df = pd.merge(hr_combined_data,eda_combined_data,how='outer', on='time')
final_df = pd.merge(final_df,temp_combined_data,how='outer', on='time')
hr_combined_data.shape,eda_combined_data.shape,temp_combined_data.shape
final_df
# sorting the table based on index
final_df = final_df.sort_index()
# dropping any null values
final_df = final_df.dropna()
# selecting only necessary colmns
final_df= final_df[['heart_rate','eda','temp','time']]
e =final_df[['time']].values
e[0][0]
time_logs
final_df
# labeling the stress indusing sessions as 1 and 0 for rest and baseline periosds
lable_list=[]
k =0
lable = 0
for i in e:
    if k <6:
        if str(i[0]) == time_logs[k]:
            # print('same')
            if k%2 ==0:
                lable = 1
                # print('lable',lable)
                k +=1
            else:
                lable = 0
                k +=1
    lable_list.append(lable)
    # print(i,lable)

final_df['lable']=lable_list
print(final_df.to_string())
fig, a_x = plt.subplots(figsize=(40, 6))
#parameters
y=final_df['lable']
x=final_df['time']
#marking time intervals
loc = mdates.MinuteLocator(interval=250000)
a_x.xaxis.set_major_locator(loc) # Locator for major axis only.
#plotting the graph
# a_x.plot(x,y)
a_x.plot(x.astype(str), y)
plt.xlabel('time')
plt.ylabel('ibi')

for t in time_logs:
    plt.axvline(x = t, color = 'b', label = 'axvline - full height')

#the intervals with stressed period are marked 1 and the rest are marked 0.
#the summary of data when the individual is stressd.
final_df[final_df['lable']==1].describe()
final_df[final_df['lable']==1].mean()
#the summary of data when the individual is not stressd.

final_df[final_df['lable']==0].describe()

# it can be observed here that the heart rate, eda and temp values are higher during streesed period compared to the rest and baselineperiod.
# checkin for correlationg between the features
cor = final_df.corr()
cor
final_df[['heart_rate','eda','temp']].values
# Machine Learning Model
# importing the required libraries
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, accuracy_score
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# splitting the data for training and testing purpose
X_train, X_test, y_train, y_test = train_test_split(
    final_df[['heart_rate','eda','temp']].values,final_df['lable'].values , test_size=0.3,random_state=54)
# standardising the data for easy computation
sc = StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)
# creating a pipeline.
pipe = Pipeline([('scaler', StandardScaler()), ('lr', RandomForestClassifier()),])
pipe.fit(X_train, y_train)
pipe.score(X_test, y_test)
# Making predicitons
y_pred =pipe.predict(X_test)
# Checking its performance
print('confusion matrix',confusion_matrix(y_test,y_pred))
print('accuracy score', accuracy_score(y_test,y_pred))
# it can be observed that the accuracy_score is consistantly above 80 percent