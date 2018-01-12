import tensorflow as tf

## load corpus
dfile = "./corpus/english_context.txt"

def show_dataset(ds):
    it = ds.make_one_shot_iterator()
    next_element = it.get_next()
    with tf.Session() as sess:
        while True:
            try:

                print(sess.run(next_element))
            except Exception as e:
                # print(e)
                print("----")
                break

ds = tf.data.TextLineDataset(dfile)
bos = "bos_"
eos = "_eos"
ds = ds.map(lambda x: tf.py_func(lambda x: x.lower(), [x], tf.string, stateful=False))
ds = ds.map(lambda x: tf.constant(bos+" ") + x + tf.constant(" "+eos))

# ds = ds.apply(tf.contrib.data.group_by_window(key_func=lambda x: x//2, \
#         reduce_func=lambda x, els: els.batch(2), window_size=2))
show_dataset(ds)

count = 20
ds2 = tf.data.Dataset().range(count)
ds2 = tf.data.Dataset().zip((ds2,ds))
show_dataset(ds2)

ds2 = ds2.apply(tf.contrib.data.group_by_window(key_func=lambda x,s: x//2,\
        reduce_func=lambda x, els: els.batch(2), window_size=2))
ds2 = ds2.map(lambda _,y: (y[0],y[1]))
show_dataset(ds2)

ds2 = tf.data.Dataset().range(count)
ds2 = tf.data.Dataset().zip((ds2, ds))
ds2 = ds2.filter(lambda x, _: x > 0)
show_dataset(ds2)

ds = tf.data.Dataset().range(100)
ds = ds.take(10)
show_dataset(ds)
ds = ds.skip(2)
show_dataset(ds)

ds = tf.data.TextLineDataset(dfile)
bos = "bos_"
eos = "_eos"
ds = ds.map(lambda x: tf.py_func(lambda x: x.lower(), [x], tf.string, stateful=False))
ds = ds.map(lambda x: tf.constant(bos+" ") + x + tf.constant(" "+eos))
ds1 = ds.take(count-1)
ds2 = ds.skip(1)
ds3 = tf.data.Dataset.zip((ds1, ds2))
show_dataset(ds3)
