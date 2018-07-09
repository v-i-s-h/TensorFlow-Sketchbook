# Multivariate prediction using LSTM
# Source: https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/

from pandas import read_csv, DataFrame, concat
from datetime import datetime
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# --------------------- Data Cleansing -----------------------------------------------------
# define parser for date
def parse( x ):
    return datetime.strptime( x, "%Y %m %d %H" )
# Load data
dataset = read_csv( 'PRSA_data_2010.1.1-2014.12.31.csv',
                        parse_dates = [['year','month','day','hour']], index_col = 0, date_parser = parse )
# Drop first column
dataset.drop( 'No', axis = 1, inplace = True )
# Manually specify column names
dataset.columns = [ 'pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain' ]
# and index column also
dataset.index.name  = 'date'
# Mark all N/As as 0
dataset['pollution'].fillna( 0, inplace = True )
# Drop first 24 as they are N/A
dataset = dataset[24:]
print( dataset.head(5) )
# ------------------------------------------------------------------------------------------

# ------------------ Visualize Data --------------------------------------------------------
# values = dataset.values
# # Specify columns to plot
# groups  = [ 0, 1, 2, 3, 5, 6, 7 ]
# i = 1
# # Plot each column
# pyplot.figure()
# for group in groups:
#     pyplot.subplot( len(groups), 1, i )
#     pyplot.plot( values[:,group] )
#     pyplot.title( dataset.columns[group], y = 0.5, loc = 'right' )
#     i += 1
# pyplot.show()
# ------------------------------------------------------------------------------------------

# ------------------ LSTM Data Preparation -------------------------------------------------
def series_to_supervised( data, n_in = 1, n_out = 1, dropnan = True ):
    n_vars  = 1 if type(data) is list else data.shape[1]
    df  = DataFrame( data )
    cols, names = list(), list()
    # Input sequence ( t-n,...t-1 )
    for i in range( n_in, 0, -1):
        cols.append( df.shift(i) )
        names += [ ('var%d(t-%d)' %(j+1,i)) for j in range(n_vars) ]
    # Forecast sequence (t,t+1,t+2,...,t+n)
    for i in range( 0, n_out ):
        cols.append( df.shift(-i) )
        if i == 0:
            names += [ ('var%d(t)' %(j+1)) for j in range(n_vars) ]
        else:
            names += [ ('var%d(t+%d)' %(j+1,i)) for j in range(n_vars) ]
    # Put it all together
    agg = concat( cols, axis = 1 )
    agg.columns = names
    if dropnan:
        agg.dropna( inplace = True )

    return agg

values  = dataset.values
# Integer encode wind direction
encoder = LabelEncoder()
values[:,4] = encoder.fit_transform( values[:,4] )
# Encsure all data is float32
values  = values.astype( 'float32' )
# Normalize features
scaler  = MinMaxScaler( feature_range = (0,1) )
scaled  = scaler.fit_transform( values )
# Frame as supervised learning
reframed    = series_to_supervised( scaled, 1, 1 )
# Drop columns we don't want to predict
reframed.drop( reframed.columns[[9,10,11,12,13,14,15]], axis = 1, inplace = True )
print( reframed.head() )
# ------------------------------------------------------------------------------------------

# ------------------- Define & Fit Model ---------------------------------------------------
values  = reframed.values
n_train_hours   = 365*24
train   = values[ :n_train_hours, : ]
test    = values[ n_train_hours:, : ]
# Split input and output
train_X, train_Y    = train[:,:-1], train[:,-1]
test_X, test_Y      = test[:,:-1], test[:,-1]
# Reshape input to be 3D [ samples, timesteps, features ]
train_X = train_X.reshape( train_X.shape[0], 1, train_X.shape[1] )
test_X  = test_X.reshape( test_X.shape[0], 1, test_X.shape[1] )
print( train_X.shape, train_Y.shape, test_X.shape, test_Y.shape )
# ------------------------------------------------------------------------------------------