import tensorflow as tf
import corpusAgent as CA

ca = CA.CorpusAgent("corpus/english_context.txt", "corpus/vocab.txt")
print("ca created")
leng = ca.dataset_len()
print(leng)

ca.gen_vocab()
sess = tf.Session()
bth = ca.gen_batch(1)
ca.show_dataset(bth)
it = bth.make_initializable_iterator()
next_element = it.get_next()
## -----embedding----- ##
src_vocat_size = leng
embedding_size = 3

embedding_encoder = tf.get_variable("embedding_encoder", [src_vocat_size, embedding_size])
embedded_word_ids = tf.nn.embedding_lookup(embedding_encoder, next_element)
sess = tf.Session()
sess.run(tf.tables_initializer())
sess.run(tf.global_variables_initializer())
sess.run(it.initializer)
print(sess.run(embedded_word_ids))

## -----encoder----- ##
num_units = 10
encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)
encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
    encoder_cell,
    embedded_word_ids,
)

print()

print("over")
