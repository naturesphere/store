import tensorflow as tf

## load corpus
dfile = "./corpus/english_context.txt"

ds = tf.data.TextLineDataset(dfile)
bos = "bos_"
eos = "_eos"
ds = ds.map(lambda x: tf.py_func(lambda x: x.lower(), [x], tf.string, stateful=False))
ds = ds.map(lambda x: tf.constant(bos+" ") + x + tf.constant(" "+eos))
ds = ds.map(lambda x: tf.string_split([x]).values)

it = ds.make_one_shot_iterator()
next_element = it.get_next()
with tf.Session() as sess:
    count = 0
    while True:
        try:
            sess.run(next_element)
            count += 1
        except Exception as e:
            # print(e)
            print("---- count: {} ----".format(count))
            break

ds1 = ds.take(count-1)
ds2 = ds.skip(1)
ds3 = tf.data.Dataset.zip((ds1, ds2))

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

show_dataset(ds3)

# ## convert to ids
# filename = "./corpus/vocab.txt"
# features = tf.constant(["bos_", "when", "is", "the", "moon", "ask", "wine", \
#                         "blue", "sky.", "_eos"])
# table = tf.contrib.lookup.index_table_from_file(vocabulary_file=filename)
# ids = table.lookup(features)
# with tf.Session() as sess:
#     sess.run(tf.tables_initializer())
#     print(sess.run(ids))


# ds4 = ds3.map(lambda x, y: (tf.cast(table.lookup(x), tf.int32), \
#                              tf.cast(table.lookup(y), tf.int32)))

# show_dataset(ds4)

# ds5 = ds4.batch(1)
# it = ds5.make_initializable_iterator()
# next_element = it.get_next()
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     sess.run(tf.tables_initializer())
#     sess.run(it.initializer)
#     while True:
#         try:
#             print(sess.run(next_element))
#         except Exception as e:
#             print(e)
#             print("----")
#             break

ds6 = ds3.padded_batch(
    batch_size = 3,
    padded_shapes = (tf.TensorShape([None]), tf.TensorShape([None]))
)

show_dataset(ds6)