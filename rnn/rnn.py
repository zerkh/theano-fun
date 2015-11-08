import theano
import theano.tensor as T
import numpy as np
from theano.tensor.basic import ivector, scalar

class RNN:
    
    def __init__(self, word_dim, hidden_dim = 100):
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        
        U = np.random.uniform(-np.sqrt(1/word_dim), np.sqrt(1/word_dim), 
                              (hidden_dim, word_dim))
        V = np.random.uniform(-np.sqrt(1/hidden_dim), np.sqrt(1/hidden_dim),
                              (hidden_dim, hidden_dim))
        W = np.random.uniform(-np.sqrt(1/hidden_dim), np.sqrt(1/hidden_dim),
                              (word_dim, hidden_dim))
        
        self.U = theano.shared(name="U", value=U.astype(theano.config.floatX))
        self.V = theano.shared(name="V", value=V.astype(theano.config.floatX))
        self.W = theano.shared(name="W", value=W.astype(theano.config.floatX))
        
        self.__build_theano__()
        
    def __build_theano__(self):
        x = ivector(name="x")
        y = ivector(name="y")
        U, V, W = self.U, self.V, self.W
        
        def forword_prop_step(x_t, s_t_prev, U, V, W):
            s_t = T.tanh(U[:,x_t] + V.dot(s_t_prev))
            o_t = T.nnet.softmax(W.dot(s_t))
            return [o_t[0], s_t]
        
        [o,s], updates = theano.scan(forword_prop_step, sequences=x, 
                                     outputs_info=[None, dict(initial=T.zeros(self.hidden_dim))], 
                                     non_sequences=[U,V,W], truncate_gradient=4, strict=True)
        
        prediction = T.argmax(o, axis=1)
        o_error = T.sum(T.nnet.categorical_crossentropy(o, y))
        
        dU = T.grad(o_error, U)
        dV = T.grad(o_error, V)
        dW = T.grad(o_error, W)
        
        self.forward = theano.function([x], o)
        self.predict = theano.function([x], prediction)
        self.c_error = theano.function([x, y], o_error)
        self.bptt = theano.function([x, y], [dU, dV, dW])
        
        learning_rate = scalar(name="learning_rate")
        self.sgd_step = theano.function([x, y, learning_rate], [], 
                                        updates=[(self.U, self.U-learning_rate*dU),
                                                 (self.V, self.V-learning_rate*dV),
                                                 (self.W, self.W-learning_rate*dW)])
        
    def calc_total_error(self, X, Y):
        return np.sum([self.c_error(x,y) for x,y in zip(X,Y)])
    
    def calc_error(self, X, Y):
        size_of_set = np.sum([len(y) for y in Y])
	print size_of_set
        return self.calc_total_error(X, Y)/size_of_set
