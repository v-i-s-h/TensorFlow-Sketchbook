# LSTM Univariate one-step predictor
# From: https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/

from pandas import DataFrame, Series, concat, read_csv, datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import sys

import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM
from math import sqrt
from matplotlib import pyplot
import numpy

# Date parsing funtion
def parser( x ):
    return datetime.strptime( '199'+x, '%Y-%m' )

# Frame a sequence of data
def timeseries_to_supervised( data, lag = 1 ):
    df  = DataFrame( data )
    columns = [ df.shift(i) for i in range(1,lag+1) ]
    columns.append( df )
    df  = concat( columns, axis = 1 )
    df.fillna( 0, inplace = True )
    return df

# Create differenced set
def difference( dataset, interval = 1 ):
    diff = list()
    for i in range( interval, len(dataset) ):
        value   = dataset[i] - dataset[i-interval]
        diff.append( value )
    return Series( diff )

# Inverted diff
def inverse_difference( history, yhat, interval = 1 ):
    return yhat + history[-interval]

# scale train and test data to [-1,1]
def scale( train, test ):
    # fit scaler
    scaler  = MinMaxScaler( feature_range = (-1,+1) )
    scaler  = scaler.fit( train )
    # Fit train
    train   = train.reshape( train.shape[0], train.shape[1] )
    train_scaled    = scaler.transform( train )
    # Fit test
    test    = test.reshape( test.shape[0], test.shape[1] )
    test_scaled     = scaler.transform( test )
    return scaler, train_scaled, test_scaled

# inverse scaling for a forecasted value
def invert_scale( scaler, X, value ):
    new_row = [ x for x in X ] + [ value ]
    array   = numpy.array( new_row )
    array   = array.reshape( 1, len(array) )
    inverted    = scaler.inverse_transform( array )
    return inverted[0,-1]

def fit_lstm( train , batch_size, nb_epoch, neurons ):
    X, y    = train[:,0:-1], train[:,-1]
    X   = X.reshape( X.shape[0], 1, X.shape[1] )
    model   = Sequential()
    model.add( LSTM(neurons,batch_input_shape=(batch_size,X.shape[1],X.shape[2]),stateful=True) )
    model.add( Dense(1) )
    model.compile( loss = 'mean_squared_error', optimizer = 'adam' )
    print( "X = {}".format(X.shape) )  
    print( "y = {}".format(y.shape) )
    print( "LSTM batch size : {}".format((batch_size,X.shape[1],X.shape[2])) )
    for i in range( nb_epoch ):
        sys.stdout.write( "\rRunning epoch: {:8d}/{}".format(i,nb_epoch) )
        sys.stdout.flush()
        model.fit( X, y, epochs = 1, batch_size = batch_size, verbose = 0, shuffle = False )
        model.reset_states()
    print("")
    return model

def forecast_lstm( model, batch_size, X ):
    X = X.reshape( 1, 1, len(X) )
    print( "Forecasting:  X:{}    batch_size:{}".format(X.shape,batch_size) )
    yhat    = model.predict( X, batch_size = batch_size )
    print( "    yhat: {} {}".format(yhat.shape,yhat) )
    return yhat[0,0]

# Load data
series  = read_csv( './shampoo.csv', header = 0, parse_dates = [0], index_col = 0,
                        squeeze = True, date_parser = parser )

# Transform data to stationary
raw_values  = series.values
diff_values = difference( raw_values, 1 )

# transform data to supervised learning
supervised  = timeseries_to_supervised( diff_values, 1 )
supervised_values   = supervised.values

train, test     = supervised_values[0:-12], supervised_values[-12:]
scaler, train_scaled, test_scaled   = scale( train, test )

# print( "train_scaled: {}".format(train_scaled) )
# print( "test_scaled: {}".format(test_scaled) )

# fit model
lstm_model  = fit_lstm( train_scaled, 1, 3000, 4 )
# Forcast the entire training set to build up state for foercasting
train_reshaped  = train_scaled[:,0].reshape( len(train_scaled), 1, 1 )
lstm_model.predict( train_reshaped, batch_size = 1 )

# Walk-forwars validation of data
predictions = list()
for i in range( len(test_scaled) ):
    # Make one step forecast
    X, y    = test_scaled[i,0:-1], test_scaled[i,-1]
    yhat    = forecast_lstm( lstm_model, 1, X )
    yhat    = invert_scale( scaler,X, yhat )
    yhat    = inverse_difference( raw_values, yhat, len(test_scaled)+1-i )
    predictions.append( yhat )
    expected    = raw_values[ len(train)+i+1 ]
    print( "Month = {:d}    Predicted = {:f}    Expected = {:f}".format(i+1,yhat,expected) )

# Report
rmse    = sqrt( mean_squared_error(raw_values[-12:],predictions) )
print( "Test RMSE : {:.3f}".format(rmse) )
