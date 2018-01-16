import tensorflow as tf
import os

class CorpusAgent:
    def __init__(self, src_file=None, vocab_file=None):
        self.bos = "bos_"
        self.eos = "_eos"
        self.src_file = src_file
        self.vocab_file = vocab_file
        self.dataset = tf.data.TextLineDataset(src_file)
        self.gen_vocab()

    def gen_vocab(self, src_file=None, vocab_file=None):
        
        if vocab_file == None:
            vocab_file = self.vocab_file
        if src_file == None:
            src_file = self.src_file

        v_set = set()
        with open(src_file,'r') as f:
            for line in f:
                ts = set(line.lower().split())
                v_set |= ts
        if os.path.exists(self.vocab_file):
            os.remove(self.vocab_file)
        
        with open(self.vocab_file, 'a', encoding="utf-8") as f:
            for v in v_set:
                f.write(v+"\n")
            f.write(self.bos+"\n")
            f.write(self.eos)

    def gen_batch(self, batch_size, src_file=None, vocab_file=None):
        if vocab_file == None:
            vocab_file = self.vocab_file
        if src_file == None:
            src_file = self.src_file

        self.dataset = self.dataset.map(lambda x: tf.py_func(lambda x: x.lower(), [x], tf.string, stateful=False))
        self.dataset = self.dataset.map(lambda x: tf.constant(self.bos+" ") + x + tf.constant(" "+self.eos))
        self.dataset = self.dataset.map(lambda x: tf.string_split([x]).values)
        leng = self.dataset_len()
        ds1 = self.dataset.take(leng-1)
        ds2 = self.dataset.skip(1)
        self.dataset = tf.data.Dataset.zip((ds1, ds2))

        ##convert to ids
        if not os.path.exists(self.vocab_file):
            self.gen_vocab(src_file,vocab_file)
        table = tf.contrib.lookup.index_table_from_file(vocabulary_file=vocab_file)        
        self.dataset = self.dataset.map(lambda x, y: (tf.cast(table.lookup(x), tf.int32), \
                                        tf.cast(table.lookup(y), tf.int32)))

        return self.dataset.padded_batch(batch_size,(tf.TensorShape([None]), tf.TensorShape([None])))


    def dataset_len(self):
        it = (self.dataset).make_one_shot_iterator()
        next_element = it.get_next()
        count = 0
        with tf.Session() as sess:
            while True:
                try:
                    sess.run(next_element)
                    count += 1
                except Exception as e:
                    # print(e)
                    break
        return count

    def show_dataset(self, dataset=None, num = 100):
        if dataset == None:
            dataset = self.dataset
        it = dataset.make_initializable_iterator()
        next_element = it.get_next()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())
            sess.run(it.initializer)
            for _ in range(num):
                try:
                    print(sess.run(next_element))
                except Exception as e:
                    # print(e)
                    print("----")
                    break
