# Import necessary libraries
import warnings
import pandas as pd
import os
from pathlib import Path
from datetime import datetime

# Import visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Import machine learning libraries
from sklearn.preprocessing import MinMaxScaler




class AssetsDataLoader:
    def __init__(self, dataset_name, split_point, begin_date=None, end_date=None, load_from_file=False):
        # Suppress warnings
        warnings.filterwarnings('ignore')

        # Store dataset name
        self.DATASERI = dataset_name

        # Define paths for data and object directories
        self.DATAPATH = os.path.join(Path(os.path.abspath(os.path.dirname(__file__))).parent, f'Data/{dataset_name}') + '/'
        self.OBJECTPATH = os.path.join(Path(os.path.abspath(os.path.dirname(__file__))).parent, 'Objects') + '/'

        # Define the CSV file name
        self.DATAFILE = dataset_name + '.csv'

        # Store split point and date range parameters
        self.split_point = split_point
        self.split_date = datetime.strptime(self.split_point, '%Y-%m-%d')
        self.begin_date = begin_date
        self.end_date = end_date

        # Load data and preprocess if not loading from file
        if not load_from_file:
            self.data  = self.load_data()  # Load and preprocess 
            #self.save_pattern()  # Save patterns to pickle file
            self.normalize_data()  # Normalize data
            self.data.to_csv(f'{self.DATAPATH}data_processed.csv', index=True)  # Save preprocessed data

            # Filter data based on date ranges
            if begin_date is not None:
                self.data = self.data[self.data.index >= begin_date]

            if end_date is not None:
                self.data = self.data[self.data.index <= end_date]

            # Split data into train and test based on split_point
            if type(split_point) == str:
                self.data.index = pd.to_datetime(self.data.index, format='%d/%m/%Y')
                self.data_train = self.data[self.data.index < self.split_date]
                self.data_test = self.data[self.data.index >= self.split_date]
            else:
                raise ValueError('Split point is not valid!')

            # Create copies of train and test data with dates
            self.data_train_with_date = self.data_train.copy()
            self.data_test_with_date = self.data_test.copy()

            # Reset indices of train and test data
            self.data_train.reset_index(drop=True, inplace=True)
            self.data_test.reset_index(drop=True, inplace=True)

        # Load processed data from file
        else:
            self.data = pd.read_csv(f'{self.DATAPATH}data_processed.csv')
            self.data.set_index('Date', inplace=True)
            self.normalize_data()  # Normalize data

            # Filter data based on date ranges
            if begin_date is not None:
                self.data = self.data[self.data.index >= begin_date]

            if end_date is not None:
                self.data = self.data[self.data.index <= end_date]

            # Split data into train and test based on split_point
            if type(split_point) == str:
                self.data.index = pd.to_datetime(self.data.index, format='%d/%m/%Y')
                self.data_train = self.data[self.data.index < self.split_date]
                self.data_test = self.data[self.data.index >= self.split_date]
            else:
                raise ValueError('Split point is not valid!')

            # Create copies of train and test data with dates
            self.data_train_with_date = self.data_train.copy()
            self.data_test_with_date = self.data_test.copy()

            # Reset indices of train and test data
            self.data_train.reset_index(drop=True, inplace=True)
            self.data_test.reset_index(drop=True, inplace=True)

    # Load and preprocess raw data
    def load_data(self):
        # Load data from CSV file
        data = pd.read_csv(f'{self.DATAPATH}{self.DATAFILE}')
        
        # Remove rows with missing values
        data.dropna(inplace=True)
        
        # Set the Date column as the index
        data.set_index('Date', inplace=True)
        
        # Rename columns for consistency
        data.rename(columns={'Close': 'close', 'Open': 'open', 'High': 'high', 'Low': 'low'}, inplace=True)
        
        # Drop unnecessary columns
        data = data.drop(['Adj Close', 'Volume'], axis=1)
        
        data['action'] = "None"
        
        return data 

    # Plot train and test data
    def plot_data(self):
        # Set figure size using seaborn
        sns.set(rc={'figure.figsize': (9, 5)})
        
        # Create pandas Series for train and test close prices with corresponding dates
        df1 = pd.Series(self.data_train_with_date.close, index=self.data.index)
        df2 = pd.Series(self.data_test_with_date.close, index=self.data.index)
        
        # Plot the train and test data along with split point
        ax = df1.plot(color='b', label='Train Data')
        df2.plot(ax=ax, color='g', label='Test Data')
        ax.axvline(x=self.split_date, color='r', linestyle='--', label='Split Point')
        ax.set(xlabel='Time', ylabel='Close Price')
        #ax.set_title(f'Train and Test sections of {self.DATASERI} dataset')
        plt.legend()
        
        # Save the plot as an image file
        plt.savefig(f'{Path(self.DATAPATH).parent}/DatasetImages/{self.DATASERI}.jpg', dpi=300)

    # Normalize data using Min-Max scaling
    def normalize_data(self):
        min_max_scaler = MinMaxScaler()
        self.data['open_norm'] = min_max_scaler.fit_transform(self.data.open.values.reshape(-1, 1))
        self.data['high_norm'] = min_max_scaler.fit_transform(self.data.high.values.reshape(-1, 1))
        self.data['low_norm'] = min_max_scaler.fit_transform(self.data.low.values.reshape(-1, 1))
        self.data['close_norm'] = min_max_scaler.fit_transform(self.data.close.values.reshape(-1, 1))
