import tensorflow as tf

ds = tf.data.Dataset.range(100)
dsb = ds.batch(17)
it = dsb.make_one_shot_iterator()
next_element = it.get_next()
with tf.Session() as sess:
    while True:
        try:
            print(sess.run(next_element))
        except Exception as e:
            # print(e)
            break