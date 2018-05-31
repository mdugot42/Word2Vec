import tensorflow as tf
import numpy as np
from tqdm import tqdm

class NeuralNetwork:

    def __init__(self, data):

        tf.reset_default_graph()

        self.hiddenLayer = 30
        self.negativeSampleSize = 10
        self.numberWords = len(data.dictionary)
        tmp = np.zeros(self.negativeSampleSize + 1)
        tmp[-1] = 1


        self.winit = tf.contrib.layers.xavier_initializer()
        self.binit = tf.constant_initializer(0)
        
        self.inputWord = tf.placeholder(tf.int32, [])
        self.labelWord = tf.placeholder(tf.int32, [])
        self.negativeSample = tf.placeholder(tf.int32, [self.negativeSampleSize])

        self.labels = tf.constant(tmp, dtype=tf.float64, shape=[self.negativeSampleSize+1])
        self.sample = tf.concat([self.negativeSample, [self.labelWord]], axis = 0)

        self.w1 = tf.get_variable("w1", shape=[self.numberWords, self.hiddenLayer], dtype=tf.float64, initializer = self.winit)
        self.b1 = tf.get_variable("b1", shape=[self.hiddenLayer], dtype=tf.float64, initializer = self.binit)
        self.w2 = tf.get_variable("w2", shape=[self.hiddenLayer, self.numberWords], dtype=tf.float64, initializer = self.winit)
        self.b2 = tf.get_variable("b2", shape=[self.numberWords], dtype=tf.float64, initializer = self.binit)

        self.gw1 = tf.gather(self.w1, [self.inputWord], axis=0)
        self.gw2 = tf.gather(self.w2, self.sample, axis=1)
        self.gb2 = tf.gather(self.b2, self.sample, axis=0)

        self.a1 = self.gw1 + self.b1;
        self.a2 = tf.matmul(self.a1, self.gw2) + self.gb2
        self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.labels, logits=self.a2)
        self.optimizer = tf.train.AdamOptimizer(0.01).minimize(self.loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def train(self, D, iterations = 200, window = 3):
        cost = 0
        for i in tqdm(range(iterations)):
            negatives = np.random.choice(range(len(D.dictionary)), self.negativeSampleSize)
            batch = D.batch(window)
            inputId = batch["id"]["input"]
            labelId = batch["id"]["output"]
            while inputId in negatives or labelId in negatives:
                negatives = np.random.choice(range(len(D.dictionary)), self.negativeSampleSize)

            _, c = self.sess.run([self.optimizer, self.loss], feed_dict={self.inputWord:inputId, self.labelWord:labelId, self.negativeSample:negatives})
            cost += c

        cost /= iterations
        print("cost : " + str(cost))

    def createVectDict(self, D):
        vdict = {}
        print("create vectors dictionary")
        for word in tqdm(D.idict):
            v = self.sess.run(self.a1, feed_dict={self.inputWord:D.dictionary[word]})
            vdict[word] = v[0]
        return vdict
            
