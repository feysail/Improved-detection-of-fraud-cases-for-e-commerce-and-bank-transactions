import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def load_data(path):
    """load data"""
    return pd.read_csv(path)


def univariate_visualize_fraud(data_frame):
    
    """univarate analysis to see the distribution in each column"""
    
    data_frame_columns = ['purchase_value', 'age', 'ip_address', 'class']
    
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()  

    
    for i, col in enumerate(data_frame_columns):
        if i < len(axes): 
            axes[i].hist(data_frame[col], bins=30, alpha=0.7, color='blue')
            axes[i].set_title(col.upper())
            axes[i].set_xlabel('Values')
            axes[i].set_ylabel('Frequency')
        else:
            break  
        
 
    plt.tight_layout()
    plt.show()
    
    
    
def univariate_visualize_credit(data_frame):
    
    """univarate analysis to see the distribution in each column"""
    
    data_frame_columns = data_frame.describe().columns
    fig, axes = plt.subplots(8, 4, figsize=(25, 15))
    axes = axes.flatten()  

    
    for i, col in enumerate(data_frame_columns):
        if i < len(axes): 
            axes[i].hist(data_frame[col], bins=30, alpha=0.7, color='red')
            axes[i].set_title(col.upper())
            axes[i].set_xlabel('Values')
            axes[i].set_ylabel('Frequency')
        else:
            break  
        
 
    plt.tight_layout()
    plt.show()
    
    

def age_purchase(data):
    plt.figure(figsize=(16, 12))
    plt.subplot(2, 2, 1)
    plt.scatter(data['age'], data['purchase_value'], alpha=0.5, color='blue')
    plt.title('Age vs Purchase Value')
    plt.xlabel('Age')
    plt.ylabel('Purchase Value')
    plt.grid()

def class_purchase(data):
    plt.subplot(2, 2, 2)
    sns.boxplot(x='class', y='purchase_value', data=data)
    plt.title('Class vs Purchase Value')
    plt.xlabel('Class')
    plt.ylabel('Purchase Value')

def class_age(data):
    plt.subplot(2, 2, 3)
    sns.boxplot(x='class', y='age', data=data)
    plt.title('Class vs Age')
    plt.xlabel('Class')
    plt.ylabel('Age')



# Adjust layout
plt.tight_layout()
plt.show()


def bivarate(data):
 """bivarate analysis"""
 plt.figure(figsize=(16, 12))
 plt.subplot(2, 2, 1)
 plt.scatter(data['age'], data['purchase_value'], alpha=0.5, color='blue')
 plt.title('Age vs Purchase Value')
 plt.xlabel('Age')
 plt.ylabel('Purchase Value')
 plt.grid()


 plt.subplot(2, 2, 2)
 sns.boxplot(x='class', y='purchase_value', data=data)
 plt.title('Class vs Purchase Value')
 plt.xlabel('Class')
 plt.ylabel('Purchase Value')


 plt.subplot(2, 2, 3)
 sns.boxplot(x='class', y='age', data=data)
 plt.title('Class vs Age')
 plt.xlabel('Class')
 plt.ylabel('Age')


 plt.subplot(2, 2, 4)
 sns.countplot(x='ip_address', hue='class', data=data, palette='viridis')
 plt.title('IP Address vs Class')
 plt.xticks(rotation=90)  

 plt.tight_layout()
 plt.show()
 
 
def per(data1,data2):
    
    data1['day_of_the_week']=data1['purchase_time'].dt.dayofweek
    
    plt.subplot(1,2,1)
    data1.groupby('day_of_the_week')['purchase_value'].sum().plot(kind='line')
    plt.subplot(1,2,2)
    data2.groupby('Time')[['Amount']].sum().plot(kind='line',color='red')
    plt.set_xticks(range(7))
    plt.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    
    

def per(data1, data2):

    
    data1['day_of_the_week'] = data1['purchase_time'].dt.dayofweek
    
    plt.figure(figsize=(15,5))
    plt.subplot(1, 2, 1)
    data1.groupby('day_of_the_week')['purchase_value'].sum().plot(kind='line', marker='o')
    plt.title('Weekly Purchase Value')
    plt.xlabel('Day of the Week')
    plt.ylabel('Total Purchase Value')
    plt.xticks(range(7), ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])

    plt.subplot(1, 2, 2)
    data2.groupby('Time')['Amount'].sum().plot(kind='line', color='red', marker='x')
    plt.title('Weekly Credit Card Amount')
    plt.ylabel('Total Amount')
    plt.xticks(range(7), ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])


    plt.tight_layout()
    plt.show()

    
    

def standard_and_normalize(data):
    """normalization narrow the range in b/n 0 and 1 so decrease the distance metrics 
          standardization set the mean to 0 and standard deviation to 1  """
          
          
    data1=data.copy()
    numeric_columns=['purchase_value','age']
    
    min_max_scaler=MinMaxScaler()
    data1[numeric_columns]=min_max_scaler.fit_transform(data1[numeric_columns])
    
    data2=data.copy()
    standard_scaler=StandardScaler()
    data2[numeric_columns]=standard_scaler.fit_transform(data1[numeric_columns])
    df=data.copy()
    df['purchase_value']=data1['purchase_value']
    df['age']=data1['age']
    return df


def check_normality_and_standardize(column1,column2):
    plt.figure(figsize=(15,5))
    plt.subplot(1,2,1)
    sns.histplot(column1,kde=True)
    plt.subplot(1,2,2)
    sns.histplot(column2,kde=True,color='red')
    
    
    
def normalize_and_standard(data):
    data1=data.copy()
    numeric_columns=data.describe().columns.to_list()
    numeric_columns.remove('Class')
    minmaxscaler=MinMaxScaler()
    data1[numeric_columns]=minmaxscaler.fit_transform(data1[numeric_columns])
    scaler=StandardScaler()
    data1[numeric_columns]=scaler.fit_transform(data1[numeric_columns])
    data[numeric_columns]=data1[numeric_columns]
    return data

    

   

    
    
     
     
     
     
     
