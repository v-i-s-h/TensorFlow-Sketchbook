# Source: https://github.com/jaungiers/LSTM-Neural-Network-for-Time-Series-Prediction/blob/master/lstm.py

import os, time, warnings
import numpy as np
from numpy import newaxis
from tensorflow.python.keras.layers import Dense, Activation, Dropout
from tensorflow.python.keras.layers import LSTM
from tensorflow.python.keras.models import Sequential

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# warnings.filterwarnings( "ignore" )

def load_data( filename, seq_len, normalize_window ):
    f = open( filename, 'rb' ).read()
    data    = f.decode().split('\n')

    sequence_length = seq_len + 1
    result  = []
    for index in range( len(data) - sequence_length ):
        result.append( data[index:index+sequence_length] )

    if normalize_window:
        results = normalize_windows( result )

    result  = np.array( results )

    row = int( 0.9* result.shape[0] )
    train   = result[:row,:]
    np.random.shuffle( train )
    x_train = train[:,:-1]
    y_train = train[:,-1]
    x_test  = result[row:,:-1]
    y_test  = result[row:,-1]

    x_train = np.reshape( x_train, (x_train.shape[0],x_train.shape[1],1) )
    x_test  = np.reshape( x_test, (x_test.shape[0],x_test.shape[1],1) )

    return [ x_train, y_train, x_test, y_test ]

def normalize_windows( window_data ):
    normalised_data = []
    for window in window_data:
        normalised_window   = [ ( (float(p)/float(window[0])) - 1) for p in window ]
        normalised_data.append( normalised_window )

    return normalised_data

def build_model( layers ):
    model   = Sequential()

    model.add( LSTM(
        input_shape = (layers[1],layers[0]),
        output_dim  = layers[1],
        return_sequences = True
    ))
    model.add( Dropout(0.2) )
    
    model.add( LSTM(
        layers[2],
        return_sequences = False
    ))
    model.add( Dropout(0.2) )

    model.add( Dense(
        output_dim  = layers[3]
    ))
    model.add( Activation("linear") )

    start = time.time()
    model.compile( loss = "mse", optimizer = "rmsprop" )
    print( '> Compilation Time: {}'.format(time.time()-start) )
    return model

def predict_point_by_point( model, data ):
    # Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
    predicted   = model.predict( data )
    predicted   = np.reshape( predicted, (predicted.size,) )
    return predicted

def predict_sequence_full( model, data, window_size ):
    # Shift the window by 1 new prediction each time, re-run predictions on new window
    curr_frame  = data[0]
    predicted   = []
    for i in range(len(data)):
        predicted.append( model.predict(curr_frame[newaxis,:,:])[0,0] )
        curr_frame  = curr_frame[1,:]
        curr_frame  = np.insert( curr_frame, [window_size-1], predicted[-1], axis = 0 )
    
    return predicted

def predict_sequence_multiple( model, data, window_size, prediction_len ):
    # Predict sequence of 50 steps before shifting prediction run forward by 50 steps
    prediction_seqs = []
    for i in range( int(len(data)/prediction_len) ):
        curr_frame  = data[ i*prediction_len ]
        predicted   = []
        for j in range(prediction_len):
            predicted.append( model.predict(curr_frame[newaxis,:,:])[0,0] )
            curr_frame  = curr_frame[1,:]
            curr_frame  = np.insert( curr_frame, [window_size-1], predicted[-1], axis = 0 )
        prediction_seqs.append( predicted )
    return prediction_seqs