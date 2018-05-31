import random
import numpy as np
from sklearn.manifold import TSNE
import tinysegmenter
import urllib.request
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib import rcParams
from tqdm import tqdm

    
def softmax(x):
    return np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)))

class Data:


    def __init__(self):
        segmenter = tinysegmenter.TinySegmenter()

        print("request from url")
        html = urllib.request.urlopen('https://ja.wikipedia.org/wiki/%E6%97%A5%E6%9C%AC')
        soup = BeautifulSoup(html, "html5lib")
        print("get paragraphs from html")
        paragraphs = soup.findAll(['p', 'dd'])
        text = ""

        print("get text from paragraphs")
        for p in paragraphs :
            text = text + p.text

        print("tokenize")
        self.tokens = segmenter.tokenize(text)

        print("create look-up table")
        self.dictionary = {}
        self.idict = []
        tmp = []
        self.frequences = [] 
        self.tokensId = [] 
        self.hasBuildRepresentation = False

        currentId = 0
        for t in self.tokens :
            if t not in "「」、（）。][『』":
                if t in self.dictionary :
                    self.frequences[self.dictionary[t]] = self.frequences[self.dictionary[t]] + 1
                else :
                    self.dictionary[t] = currentId
                    self.idict.append(t)
                    self.frequences.append(1)
                    currentId = currentId + 1
                tmp.append(t)
                self.tokensId.append(self.dictionary[t])
        print("use log frequences")
        for i in range(len(self.frequences)) :
            self.frequences[i] = np.log(self.frequences[i]);

        self.tokens = tmp

        #self.frequences = softmax(self.frequences)
        print("tokens probability")
        self.tokensProb = self.createProbTable(self.tokensId)
    


    def createProbTable(self, listWords):
        probs = []
        for w in listWords:
            probs.append(np.max(self.frequences) - self.frequences[w])
        return softmax(probs)

    def idFromIndex(self, index):
        ids = []
        for i in index:
            ids.append(self.tokensId[i])
        return ids

    def batch(self, window):
        #print("select random batch")
        target = np.random.choice(np.arange(0, len(self.tokens)), p = self.tokensProb)

        start = target - window
        end = target + window
        start = start if start >= 0  else 0
        end = end if end < len(self.tokens)  else len(self.tokens)
        bow = np.append( np.arange(start, target - 1), np.arange(target + 1, end) )
        bowProb = self.createProbTable(self.idFromIndex(bow))
        i = np.random.choice(bow, p=bowProb)
        return {"word":{"input":self.tokens[target], "output":self.tokens[i]}, "id":{"input":self.dictionary[self.tokens[target]], "output":self.dictionary[self.tokens[i]]}}

    def create2DDict(self, NN):
        vdict = NN.createVectDict(self)
        dict2D = {}
        tmpMatrix = []
        tmpKeys = []
        print("create temporary matrix")
        for word in tqdm(self.idict):
            tmpMatrix.append(vdict[word])
            tmpKeys.append(word)
        print("convert with TSNE")
        tmpMatrix = TSNE(n_components=2).fit_transform(np.array(tmpMatrix)) 
        print("create 2D dict")
        for i in tqdm(range(len(tmpMatrix))):
            dict2D[tmpKeys[i]] = tmpMatrix[i]
        self.hasBuildRepresentation = True
        self.vdict = vdict
        self.dict2D = dict2D
        self.matrix = tmpMatrix

    def wordCloud(self, sample = 20, xmin = -100, xmax = 100, ymin = -100, ymax = 100):
        if self.hasBuildRepresentation == False:
            print("Error - no representation : call 'create2DDict' to build 2D representation.")
            return 
        fig = plt.figure(figsize=(10, 8))
        fp = FontProperties(fname='./Osaka.ttc', size=10)
        rcParams['font.family'] = fp.get_name()
        for word in tqdm(np.random.choice(self.idict, sample)):
            point = self.dict2D[word]
            x = point[0]
            y = point[1]
            if x >= xmin and x <= xmax and y >= ymin and y <= ymax:
                plt.plot(x, y, 'bo')
                plt.text(x * (1 + 0.01), y * (1 + 0.01) , word, fontsize=12) 
        plt.xlim((xmin, xmax))
        plt.ylim((ymin, ymax))
        plt.show()
