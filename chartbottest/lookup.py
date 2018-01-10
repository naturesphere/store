import tensorflow as tf

## load corpus
dfile = "./corpus/english_context.txt"
# dfile = "./corpus/chinese_context.txt"
ds = tf.data.TextLineDataset(dfile)

ds = ds.map(lambda x: tf.py_func(lambda x: x.lower(), [x], tf.string, stateful=False))
ds = ds.map(lambda x: tf.string_split([x],delimiter=" ").values)

it = ds.make_one_shot_iterator()
next_element = it.get_next()

with tf.Session() as sess:
    while True:
        try:
            # sess.run(tf.global_variables_initializer())
            print(sess.run(next_element))
        except Exception as e:
            # print(e)
            print("---")
            break

## convert to ids
filename = "./corpus/vocab.txt"
features = tf.constant(["_bos_","when","is","the","moon","ask","wine","blue","sky.","_eos_"])
# features = tf.constant(["明","月","几","时","有","把","酒","问","青","天"])
table = tf.contrib.lookup.index_table_from_file(vocabulary_file=filename)
ids = table.lookup(features)
se = tf.Session()
se.run(tf.tables_initializer())
print(se.run(ids))


it = ds.make_one_shot_iterator()
next_element = it.get_next()
ids = table.lookup(next_element)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    while True:
        try:
            print(sess.run(ids))
        except Exception as e:
            # print(e)
            print("---")
            break

