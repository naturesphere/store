import tensorflow as tf
import corpusAgent as CA

ca = CA.CorpusAgent("corpus/english_context.txt","corpus/vocab.txt")
print("ca created")

ca.gen_vocab()

# sess = tf.Session()
bth = ca.gen_batch(10)
ca.show_dataset(bth)
print("over")