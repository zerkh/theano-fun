import theano
import theano.tensor as T
import numpy as np
from theano.tensor.basic import ivector, scalar

class LSTM:
    def __init__(self, word_dim, hidden_dim):
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        
        #input gate
        Wxi = np.random.uniform(-np.sqrt(1/word_dim), np.sqrt(1/word_dim), (hidden_dim, word_dim))
        Whi = np.random.uniform(-np.sqrt(1/hidden_dim), np.sqrt(1/hidden_dim), (hidden_dim, hidden_dim))
        Wci = np.random.uniform(-np.sqrt(1/hidden_dim), np.sqrt(1/hidden_dim), (hidden_dim, 1))
        
        #forget gate
        Wxf = np.random.uniform(-np.sqrt(1/word_dim), np.sqrt(1/word_dim), (hidden_dim, word_dim))
        Whf = np.random.uniform(-np.sqrt(1/hidden_dim), np.sqrt(1/hidden_dim), (hidden_dim, hidden_dim))
        Wcf = np.random.uniform(-np.sqrt(1/hidden_dim), np.sqrt(1/hidden_dim), (hidden_dim, 1))
        
        #cell
        Wxc = np.random.uniform(-np.sqrt(1/word_dim), np.sqrt(1/word_dim), (hidden_dim, word_dim))
        Whc = np.random.uniform(-np.sqrt(1/hidden_dim), np.sqrt(1/hidden_dim), (hidden_dim, hidden_dim))
        
        #output gate
        Wxo = np.random.uniform(-np.sqrt(1/word_dim), np.sqrt(1/word_dim), (hidden_dim, word_dim))
        Who = np.random.uniform(-np.sqrt(1/hidden_dim), np.sqrt(1/hidden_dim), (hidden_dim, hidden_dim))
        Wco = np.random.uniform(-np.sqrt(1/hidden_dim), np.sqrt(1/hidden_dim), (hidden_dim, 1))
        
        Wo = np.random.uniform(-np.sqrt(1/hidden_dim), np.sqrt(1/hidden_dim), (word_dim, hidden_dim))
        
        self.Wxi = theano.shared(name="Wxi", value=Wxi.astype(theano.config.floatX))
        self.Whi = theano.shared(name="Whi", value=Whi.astype(theano.config.floatX))
        self.Wci = theano.shared(name="Wci", value=Wci.astype(theano.config.floatX))
        
        self.Wxf = theano.shared(name="Wxf", value=Wxf.astype(theano.config.floatX))
        self.Whf = theano.shared(name="Whf", value=Whf.astype(theano.config.floatX))
        self.Wcf = theano.shared(name="Wcf", value=Wcf.astype(theano.config.floatX))
        
        self.Wxc = theano.shared(name="Wxc", value=Wxc.astype(theano.config.floatX))
        self.Whc = theano.shared(name="Whc", value=Whc.astype(theano.config.floatX))
        
        self.Wxo = theano.shared(name="Wxo", value=Wxo.astype(theano.config.floatX))
        self.Who = theano.shared(name="Who", value=Who.astype(theano.config.floatX))
        self.Wco = theano.shared(name="Wco", value=Wco.astype(theano.config.floatX))
        
        self.Wo = theano.shared(name="Wo", value=Wo.astype(theano.config.floatX))
        
        self.__build_theano__()
    
    def __build_theano__(self):
        x = ivector("x")
        y = ivector("y")
        hidden_dim = self.hidden_dim
        word_dim = self.word_dim
        
        Wxi, Whi, Wci, Wxf, Whf, Wcf, Wxc, Whc, Wxo, Who, Wco, Wo = self.Wxi, self.Whi, self.Wci, self.Wxf, self.Whf, self.Wcf, self.Wxc, self.Whc, self.Wxo, self.Who, self.Wco, self.Wo
        
        def forward_prop(x_t, c_prev_t, h_prev_t,
                         Wxi, Whi, Wci, Wxf, Whf, Wcf, Wxc, Whc, Wxo, Who, Wco, Wo):
            input_gate = T.tanh(Wxi.dot(x_t) + Whi.dot(h_prev_t) + Wci*c_prev_t)
            forget_gate = T.tanh(Wxf.dot(x_t) + Whf.dot(h_prev_t) + Wcf*c_prev_t)
            
            a_c_t = Wxc.dot(x_t) + Whc.dot(h_prev_t)
            c_t = input_gate * T.nnet.sigmoid(a_c_t) + forget_gate * c_prev_t
            
            output_gate = T.tanh(Wxo.dot(x_t) + Who.dot(h_prev_t) + Wco*c_t)
            h_t = output_gate * T.tanh(c_t)
            o_t = Wo.dot(h_t)
            
            return [o_t[0], c_t, h_t]
        
        [o, c, h] = theano.scan(forward_prop, sequences = x, 
                                outputs_info = [None, dict(initial=T.zeros(hidden_dim)), dict(initial=T.zeros(hidden_dim))],
                                non_sequences = [Wxi, Whi, Wci, Wxf, Whf, Wcf, Wxc, Whc, Wxo, Who, Wco, Wo], 
                                strict = True)
        
        prediction = T.argmax(o, axis=1)
        c_error = T.sum(T.nnet.categorical_crossentropy(o, y))
        
        dWxi = T.grad(c_error, Wxi)
        dWhi = T.grad(c_error, Whi)
        dWci = T.grad(c_error, Wci)
        dWxf = T.grad(c_error, Wxf)
        dWhf = T.grad(c_error, Whf)
        dWcf = T.grad(c_error, Wcf)
        dWxc = T.grad(c_error, Wxc)
        dWhc = T.grad(c_error, Whc)
        dWxo = T.grad(c_error, Wxo)
        dWho = T.grad(c_error, Who)
        dWco = T.grad(c_error, Wco)
        dWo = T.grad(c_error, Wo)
        
        forward = theano.function([x], o)
        predict = theano.function([x], prediction)
        
        learning_rate = scalar("learning_rate")
        
        sgd_step = theano.function([x,y], [],
                                   updates = [(self.Wxi, self.Wxi-learning_rate*dWxi),
                                              (self.Whi, self.Whi-learning_rate*dWhi),
                                              (self.Wci, self.Wci-learning_rate*dWci),
                                              (self.Wxf, self.Wxf-learning_rate*dWxf),
                                              (self.Whf, self.Whf-learning_rate*dWhf),
                                              (self.Wcf, self.Wcf-learning_rate*dWcf),
                                              (self.Wxo, self.Wxo-learning_rate*dWxo),
                                              (self.Who, self.Who-learning_rate*dWho),
                                              (self.Wco, self.Wco-learning_rate*dWco),
                                              (self.Wxc, self.Wxc-learning_rate*dWxc),
                                              (self.Whc, self.Whc-learning_rate*dWhc),
                                              (self.Wo, self.Wo-learning_rate*dWo)])
        
    def calculate_total_loss(self, X, Y):
        return np.sum([self.ce_error(x,y) for x,y in zip(X,Y)])
    
    def calculate_loss(self, X, Y):
        # Divide calculate_loss by the number of words
        num_words = np.sum([len(y) for y in Y])
        return self.calculate_total_loss(X,Y)/float(num_words)