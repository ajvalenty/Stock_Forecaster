import pandas as pd
import quandl, math, datetime
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib.pyplot import style
global df
global stock
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
global scores
#Quandl api key for access to data
quandl.ApiConfig.api_key = "y9eonNQk3vUNCjARxes6"

    
class Linear(object):
    def __init__(self, name, stock):
        self.stock = stock    
        self.name = name
        
    def __getname__(name):
        return name;
    def __getstock__(stock):
        return stock;
        
    def database(name, stock):
        if user.stock == "google" or user.stock == "googl" or user.stock == "alphabet": 
            return 'WIKI/GOOGL'
        elif user.stock == "apple" or user.stock == "aapl":
            return 'WIKI/AAPL'
        elif user.stock == "tesla" or user.stock == "tsla":
            return 'WIKI/TSLA'
        elif user.stock == "amazon" or user.stock == "amzn":
            return 'WIKI/AMZN'
        elif user.stock == "dell" or user.stock == "DELL":
            return 'WIKI/AMZN'
        elif user.stock == "garmin" or user.stock == "grmn":
            return 'WIKI/GRMN'
        elif user.stock == "microsoft" or user.stock == "msft":
            return 'WIKI/MSFT'
        elif user.stock == "nike" or user.stock == "nke":
            return 'WIKI/NKE'
        elif user.stock == "under armour" or user.stock == "uaa":
            return 'WIKI/UAA'
        elif user.stock == "intel" or user.stock == "intc":
            return 'WIKI/INTC'
    
        
        
    
   # def time():
      #  0.01, 1 month
      #  .1, 1 year
    def calc(df,stock):
        df = quandl.get(user.database(stock))
        df = df.copy()
    
        #Grabs the important factors in stocks from Quandl 
        df = df[["Adj. Open", "Adj. High", "Adj. Low", "Adj. Close", "Adj. Volume"]]

        #Creates a new factor: the high - low percentage
        df["HL_PCT"] = (df["Adj. High"] - df["Adj. Low"])/df["Adj. Low"]*100.0

        #Creates a new factor: percentage change
        df["PCT_change"] = (df["Adj. Close"] - df["Adj. Open"])/ df["Adj. Open"] * 100.0
        
        #Adds both new factors into data
        df = df[["Adj. Close","HL_PCT","PCT_change","Adj. Volume"]]

        #Creates a new column for forecasting/predicting
        forecast_col = "Adj. Close"
        df.fillna(-99999, inplace=True)
        forecast_out = int(math.ceil(.01*len(df)))

        df["label"] = df[forecast_col].shift(-forecast_out)
    
        X = np.array(df.drop(["label"],1))
        X = preprocessing.scale(X)
        X_lately = X[-forecast_out:]
        X = X[:-forecast_out]
    
        #Drops any data that is NaN
        df.dropna(inplace=True)
        y = np.array(df["label"])


        #runs a train test split: splits the data- 80% for training, 20% for testing
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=.2)

        #Linear Regressions chosen as the most effective machine learning algorithm
        clf = LinearRegression(n_jobs=-1)
        clf.fit(X_train, y_train)
    
        #Calculates accuracy by scoring it
        accuracy = clf.score(X_test, y_test)
        print("The linear regression's accuracy is...")
        print(accuracy)
        print("Graphing...")
    
        #Creates forecast set to predict
        forecast_set = clf.predict(X_lately)
        df["Forecast"] = np.nan
    
        #Finds the last location in array
        last_date = df.iloc[-1].name
        last_unix = last_date.timestamp()
    
        #Seconds in a day
        one_day = 86400
    
        #Finds the next day
        next_unix = last_unix + one_day
    
        #Iterating through the forecast set
        #Setting each forecast as the values in the dataframe
        for i in forecast_set:
            next_date = datetime.datetime.fromtimestamp(next_unix)
            next_unix += one_day
            df.loc[next_date] = [np.nan for __ in range(len(df.columns)-1)] + [i]
            
        ff = df

        user.graph(df,stock)
        user.compare(ff,stock)
        
    def graph(self,df,stock):
        #Style for graph set as ggplot
        style.use("ggplot")
    
        #Graphing adj. close and actual forecast
        #Creates a legend and labels for graph
     
        df["Adj. Close"].plot()
        df["Forecast"].plot()
        plt.legend(loc=4)
        plt.title(user.__getname__())
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.show()
            
            
    def classify(self, stock): 
        if user.stock == "google" or user.stock == "googl" or user.stock == "alphabet": 
          print("You chose Google! Calculating and graphing")
          df = quandl.get('WIKI/GOOGL')
          stock = "Google Stock with Prediction"
          user.calc(df)  
          #graph(df,stock)
          
        elif user.stock == "apple" or user.stock == "aapl": 
          print("You chose Apple! Calculating and graphing...")
          df = quandl.get('WIKI/AAPL')
          stock = "Apple Stock with Prediction";
          user.calc(df)
          #graph(df,stock)
          
        elif user.stock == "tesla" or user.stock == "tsla":
          print("You chose Tesla! Calculating and graphing...")
          df = quandl.get('WIKI/TSLA')
          stock = "Tesla Stock with Prediction"
          user.calc(df)
          #graph(df,stock)
          
        elif user.stock == "amazon" or user.stock == "amzn":
          print("You chose Amazon! Calculating and graphing...")
          df = quandl.get('WIKI/AMZN')
          stock = "Amazon Stock with Prediction"
          user.calc(df)
          #user.graph(df)
          
        elif user.stock == "dell":
          print("You chose Dell! Calculating and graphing...")
          df = quandl.get('WIKI/DELL')
          stock = "Dell Stock with Prediction"
          user.calc(df)
          #user.graph(df)
          
        elif user.stock == "microsoft" or user.stock == "msft":
          print("You chose Microsoft! Calculating and graphing...")
          df = quandl.get('WIKI/MSFT')
          stock = "Microsoft Stock with Prediction"
          user.calc(df)
          #user.graph(df)
          
        elif user.stock == "nike" or user.stock == "nke": 
          print("You chose Nike! Calculating and graphing...")
          df = quandl.get('WIKI/NKE')
          stock = "Nike Stock with Prediction";
          user.calc(df)
          
        elif user.stock == "under armour" or user.stock == "uaa": 
          print("You chose Under Armour! Calculating and graphing...")
          df = quandl.get('WIKI/UAA')
          stock = "Under Armor Stock with Prediction";
          user.calc(df)
          
        elif user.stock == "intel" or user.stock == "intc": 
          print("You chose Intel! Calculating and graphing...")
          df = quandl.get('WIKI/INTC')
          stock = "Intel Stock with Prediction";
          user.calc(df,stock)

          
        
    def compare(self, ff, stock):
        print("Do you want to zoom in?")
        days = input("If so, how many days would you like to see on the x-axis? (The forecast is 30 days) ")
        days = int(days)
        
        style.use("ggplot")
        
        ff["Adj. Close"][-days:-1].plot()
        ff["Forecast"][-days:-1].plot()
        
        plt.legend(loc=4)
        plt.title(user.__getname__())
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.show()

class SequentialModel(object):
    def __init__(self):
  
    
     def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
       n_vars = 1 if type(data) is list else data.shape[1]
       df = DataFrame(data)
       cols, names = list(), list()
       for i in range(n_in, 0, -1):
           cols.append(df.shift(i))
           names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]

       for i in range(0, n_out):
           cols.append(df.shift(-i))
           if i == 0:
               names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
           else:
               names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]

       agg = concat(cols, axis=1)
       agg.columns = names
       if dropnan:
           agg.dropna(inplace=True)
       return agg
   
    def execute(object):    
       dataset = read_csv('datafinal.csv', header=0, index_col=0)

       values = dataset.values
       encoder = LabelEncoder()
       values[:,4] = encoder.fit_transform(values[:,4])

       values = values.astype('float32')

       scaler = MinMaxScaler(feature_range=(0, 1))

       scaled = scaler.fit_transform(values)
       reframed = series_to_supervised(scaled, 1, 1)

       reframed.drop(reframed.columns[[2,3,4,5]], axis=1, inplace=True)

       print(reframed.head())

       values = reframed.values                                                                                                                                                    

       n_train_hours = 365 * 24
       train = values[:n_train_hours, :]
       test = values[n_train_hours:, :]
       train_X, train_y = train[:, :-1], train[:, -1]
       test_X, test_y = test[:, :-1], test[:, -1]
       train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
       test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
       print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

       model = Sequential()
       model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
       model.add(Dense(1))
       model.compile(loss='mae', optimizer='adam')
#setting up our loss function in our case we want the lwest loss value and the highest accuracy given an epoch                                                                                              

       
       history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)

       print(history.history.keys())
    # summarize history for accuracy
       plt.plot(history.history['acc'])
       plt.plot(history.history['val_acc'])  # RAISE ERROR
       plt.title('model accuracy')
       plt.ylabel('accuracy')
       plt.xlabel('epoch')
       plt.legend(['train', 'test'], loc='upper left')
       plt.show()
    # summarize history for loss
       plt.plot(history.history['loss'])
       plt.plot(history.history['val_loss']) #RAISE ERROR
       plt.title('model loss')
       plt.ylabel('loss')
       plt.xlabel('epoch')
       plt.legend(['train', 'test'], loc='upper left')
       plt.show()
#all the input calls   


name_input = input("What is your name? ")
print("Welcome to  " + name_input + "'s stock predictor")

print()
print("This stock predictor is to compare the difference between a linear regression model and a LSTM model")
model_input = input("Which model would you like to use? (L for Linear Regression and LSTM for the Sequential Model) ")

if model_input =="L":
   stock_input = input("Enter the stock name or company name you would like to predict: ").lower()

   user = Linear(name_input, stock_input)  
   user.classify(stock_input)
   
else: 
   user = SequentialModel()
   user.execute()






