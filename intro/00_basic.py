import tensorflow as tf

x   = tf.constant( 2.0 )

sess    = tf.Session()
print( x )
print( sess.run(x) )

z   = tf.placeholder( tf.float32 )
comp    = tf.add( z, x )

print( sess.run(comp,feed_dict={z:13.0}) )
print( z )
print( sess.run(z,feed_dict={z:12.99}))
print( comp )
# print( sess.run(comp) )   # ERROR