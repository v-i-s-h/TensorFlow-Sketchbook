# Uncertainity using drop out

import matplotlib.pyplot as plt

from Dataset import *

ds = MNISTDataset(6)

f, a = plt.subplots( 8, 8 )

b1X, b1Y = ds.get_train_batch( 8*8 )
print( b1X.shape, b1Y.shape )

print( np.unique(b1Y) )

for (i,ax1) in enumerate(a):
    for (j,ax2) in enumerate(ax1):
        ax2.imshow( np.squeeze(b1X[i*8+j,],axis=2) )

plt.show()