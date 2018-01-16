import tensorflow as tf
import numpy as np

print(tf.__version__)

def show_dataset(ds):
    it = ds.make_initializable_iterator()
    next_element = it.get_next()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        sess.run(it.initializer)
        while True:
            try:
                print(sess.run(next_element))
            except Exception as e:
                # print(e)
                print("----")
                break
'''
ds = tf.data.Dataset().range(100)
show_dataset(ds)
dsb = ds.batch(10)
show_dataset(dsb)

## y=2x+3
# x = tf.constant(10)
w = tf.constant(2,dtype=tf.int64)
b = tf.constant(3,dtype=tf.int64)

print(w)
it = ds.make_one_shot_iterator()
x = it.get_next()
y = w*x+b

with tf.Session() as sess:
    while True:
        try:
            print(sess.run(y))
        except tf.errors.OutOfRangeError:
            break
'''
#data
x = np.arange(100)
print(x)
y = 2*x+3
print(y)
#net
#w*x+b
w = tf.get_variable('w',tf.float32,initializer=tf.constant_initializer(1))
b = tf.get_variable('b',tf.float32,initializer=tf.constant_initializer(1))
xi = tf.placeholder(tf.float32)
yl = tf.placeholder(tf.float32)
yh = w*xi + b
#controller
loss = tf.reduce_mean((yl-yh)**2)
opt = tf.train.AdamOptimizer(1.0).minimize(loss)
# opt = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
#train
se = tf.Session()
se.run(tf.global_variables_initializer())

for _ in range(500):
    print("w: " + str(se.run(w)))
    print("b: " + str(se.run(b)))
    se.run([opt,loss],feed_dict={xi:x,yl:y})

# for _ in range(100):
#     se.run(tf.global_variables_initializer())
#     # print("loss: "+str(se.run(loss,feed_dict={xi:x})))
#     print("w: " + str(se.run(w)))
#     print("b: " + str(se.run(b)))
#     # print(se.run(yh,feed_dict={xi:x}))
#     se.run(opt, feed_dict={xi:x,yl:y})
#test
