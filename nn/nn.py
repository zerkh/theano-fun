import theano.tensor as T
from numpy import *
import theano
from theano.tensor.basic import dvector
from datetime import datetime
import sys

class NN:
    
    def __init__(self, input_dim, hidden_dim=50):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        W1 = random.uniform(-1, 1, (input_dim, hidden_dim))
        b1 = zeros((1, hidden_dim))
        W2 = random.uniform(-1, 1, (hidden_dim, 3))
        b2 = zeros((1, 3))
        
        self.W1 = theano.shared(name="W1", value=W1.astype(theano.config.floatX))
        self.W2 = theano.shared(name="W2", value=W2.astype(theano.config.floatX))
        self.b1 = theano.shared(name="b1", value=b1.astype(theano.config.floatX))
        self.b2 = theano.shared(name="b2", value=b2.astype(theano.config.floatX))
        
        self.theano = {}
        self.__theano_build__()
        
    def __theano_build__(self):
        W1 = self.W1
        W2 = self.W2
        b1 = self.b1
        b2 = self.b2
        x = dvector(name="x")
        y = dvector(name="y")
        def forward(x):
            h = T.tanh(x.dot(W1) + b1)
            o = T.nnet.softmax(h.dot(W2) + b2)
            return o
        
        o = forward(x)
        prediction = T.argmax(o, axis=1)
        o_error = ((y-o)**2).sum()
        
        dW1 = T.grad(o_error, W1)
        dW2 = T.grad(o_error, W2)
        db1 = T.grad(o_error, b1)
        db2 = T.grad(o_error, b2)
        
        self.forward_propagation = theano.function([x], o)
        self.predict = theano.function([x], prediction)
        self.cerror = theano.function([x, y], o_error)
 
        learning_rate = T.scalar("learning_rate")
        self.sgd_step = theano.function([x,y,learning_rate], [], 
                                        updates=[(self.W1, self.W1-learning_rate*dW1),
                                                 (self.W2, self.W2-learning_rate*dW2),
                                                 (self.b1, self.b1-learning_rate*db1),
                                                 (self.b2, self.b2-learning_rate*db2)])
    def calc_total_error(self, x, y):
        return sum([self.cerror(x,y) for x,y in zip(x,y)])
    
    def calc_error(self, x, y):
        num_of_set = len(y)
        return self.calc_total_error(x, y)/num_of_set
        
def train_with_sgd(model, X_train, y_train, learning_rate=0.05, nepoch=100, evaluate_after = 5):
    num_examples_seen = 0
    
    for epoch in range(nepoch):
        if (epoch % evaluate_after == 0):
            loss = model.calc_error(X_train, y_train)
            time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            print "%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss)
            
            sys.stdout.flush()
            
        for i in range(len(y_train)):
            model.sgd_step(X_train[i], y_train[i], learning_rate)
            num_examples_seen += 1
            
def test(model, X_test, y_test):
    true_times = 0
    
    for x,y in zip(X_test, y_test):
        predict = model.predict(x)
        if y[predict] == 1.0:
            true_times += 1
        print "features:%s predict:%d true:%s" %(str(x), predict, str(y))
       
    print "Precise: %f" %(float(true_times)/len(y_test))

trainfile = open("iris.data")
testfile = open("bezdekIris.data")

dict_labels = {"Iris-setosa":0, "Iris-versicolor":1, "Iris-virginica":2}

X_train = []
y_train = []

X_test = []
y_test = []

for line in trainfile:
    seps = line.strip().split(",")
    x = seps[:-1]
    y = zeros(3)
    y[dict_labels[seps[-1]]] = 1
    X_train.append(asarray(x, dtype=theano.config.floatX))
    y_train.append(asarray(y, dtype=theano.config.floatX))
    
for line in testfile:
    seps = line.strip().split(",")
    x = seps[:-1]
    y = zeros(3)
    y[dict_labels[seps[-1]]] = 1
    X_test.append(asarray(x, dtype=theano.config.floatX))
    y_test.append(asarray(y, dtype=theano.config.floatX))


model = NN(shape(X_train)[1])
train_with_sgd(model, X_train, y_train)
test(model, X_test, y_test)
