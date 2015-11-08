import numpy as np

def softmax(x):
    xt = np.exp(x - np.max(x))
    return xt / np.sum(xt)

def save_model_parameters_theano(outfile, model):
    Wxi, Whi, Wci, Wxf, Whf, Wcf, Wxc, Whc, Wxo, Who, Wco, Wo = model.Wxi.get_value(), model.Whi.get_value(), model.Wci.get_value(),\
    model.Wxf.get_value(), model.Whf.get_value(), model.Wcf.get_value(), model.Wxc.get_value(), model.Whc.get_value(),\
    model.Wxo.get_value(), model.Who.get_value(), model.Wco.get_value()
    np.savez(outfile, Wxi=Wxi, Whi=Whi, Wci=Wci, Wxf=Wxf, Whf=Whf, Wcf=Wcf, Wxc=Wxc, Whc=Whc, Wxo=Wxo, Who=Who, Wco=Wco, Wo=Wo)
    print "Saved model parameters to %s." % outfile
   
def load_model_parameters_theano(path, model):
    npzfile = np.load(path)
    Wxi, Whi, Wci, Wxf, Whf, Wcf, Wxc, Whc, Wxo, Who, Wco, Wo = npzfile["Wxi"], npzfile["Whi"], npzfile["Wci"], npzfile["Wxf"], npzfile["Whf"],npzfile["Wcf"], npzfile["Wxc"], npzfile["Whc"], npzfile["Wxo"], npzfile["Who"],npzfile["Wco"], npzfile["Wo"]
    model.hidden_dim = Wxi.shape[0]
    model.word_dim = Wxi.shape[1]
    model.Wxi.set_value(Wxi)
    model.Whi.set_value(Whi)
    model.Wci.set_value(Wci)
    model.Wxf.set_value(Wxf)
    model.Whf.set_value(Whf)
    model.Wcf.set_value(Wcf)
    model.Wxc.set_value(Wxc)
    model.Whc.set_value(Whc)
    model.Wxo.set_value(Wxo)
    model.Who.set_value(Who)
    model.Wco.set_value(Wco)
    model.Wo.set_value(Wo)
    print "Loaded model parameters from %s. hidden_dim=%d word_dim=%d" % (path, Wxi.shape[0], Wxi.shape[1])