# Source: https://github.com/jaungiers/LSTM-Neural-Network-for-Time-Series-Prediction/blob/master/run.py

import lstm
import time
import matplotlib.pyplot as plt

def plot_results( predicted_data, true_data ):
    fig = plt.figure( facecolor = 'white' )
    ax  = fog.add_subplot(111)
    ax.plot( true_data, label = "True data" )
    plt.plot( predicted_data, label = "Predicted data" )
    plt.legend()
    plt.show()

def plot_results_multiple( predicted_data, true_data, prediction_len ):
    fig = plt.figure( facecolor = 'white' )
    ax  = fig.add_subplot( 111 )
    ax.plot( true_data, label = "True data" )
    # Pad the list of predictions to shift it in the graph to it's correct shift
    for i, data in enumerate(predicted_data):
        padding = [ None for p in range(i*prediction_len) ]
        plt.plot( padding  + data, label = "Prediction" )
        plt.legend()
    plt.show()

# Run main thread
if __name__ == '__main__':
    global_start_time   = time.time()
    epochs  = 1
    seq_len = 50

    print( "> Loading data..." )

    x_train, y_train, x_test, y_test    = lstm.load_data( 'sp500.csv', seq_len, True )

    print( " > Data Loaded. Compiling..." )

    model   = lstm.build_model( [1,50,100,1] )

    model.fit(
        x_train,
        y_train,
        batch_size  = 512,
        nb_epochs   = epochs,
        validation_split = 0.05
    )

    predictions = lstm.predict_sequences_multiple( model, X_test, seq_len, 50 )
    # predicted   = lstm.predict_sequence_full( model, x_test, seq_len )
    # predicted   = lstm.predict_point_by_point( model, X_test )

    print( 'Training duration (s): {}'.format(time.time()-global_start_time) )
    plot_results_multiple( predictions, y_test, 50 )