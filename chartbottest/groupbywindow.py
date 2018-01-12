import tensorflow as tf

ds = tf.data.Dataset.range(10)
ds = ds.apply(tf.contrib.data.group_by_window(key_func=lambda x: x//2, \
        reduce_func=lambda x, els: els.batch(2), window_size=2))
it = ds.make_one_shot_iterator()

with tf.Session() as sess:
    while True:
        try:
            print(sess.run(it.get_next()))
        except:
            print("===")
            break
