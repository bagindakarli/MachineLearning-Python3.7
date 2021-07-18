#!/usr/bin/env python
# coding: utf-8

# In[43]:


# Baginda
# 130-
# Pembelajaran Mesin (IF-41)


# In[2]:


import numpy as np
np.random.seed(213)

def affine_forward(X, W, b):
    V = np.dot(X, W) + b
    return V

def affine_backward(dout, X, W, b):
    dX = np.dot(dout, W.T)
    dW = np.dot(X.T, dout)
    db = np.sum(dout, axis=0, keepdims=True)
    return dX, dW, db

def sigmoid_forward(V):
    act = 1/(1+np.exp(-V))
    return act

def sigmoid_backward(dout, act):
    dact = act-act**2
    dout = dout*dact
    return dout


# In[47]:


# SINGLE LAYER


# In[4]:


def train_single_layer(X, y, lr, max_epoch, weights=None, history=None):
    n_data, n_dim = X.shape
    _, n_out = y.shape

    if weights is None:
        W1 = 2 * np.random.random((n_dim, n_out)) - 1
        b1 = np.zeros((1, n_out))
        history = []
    else:
        W1, b1 = weights

    for ep in range(max_epoch):
        # forward pass
        V1 = affine_forward(X, W1, b1)
        y_hat = sigmoid_forward(V1)

        # calculate loss
        E = y-y_hat
        mse = np.mean(E**2)
        history.append(mse)
        
        acc = np.sum(y==np.round(y_hat))/n_data
        print('epoch: %i/%i, mse: %.7f, acc: %.2f' % (ep, max_epoch, mse, acc))

        # backward pass
        dV1 = sigmoid_backward(E, y_hat)
        dX, dW1, db1 = affine_backward(dV1, X, W1, b1)

        # weights update
        W1 = W1 + lr*dW1
        b1 = b1 + lr*db1

    weights = (W1, b1)

    return history, weights


# In[8]:


X = np.array([ [0, 0], [0, 1], [1, 0], [1, 1] ])
y = np.array([ [0, 0, 0, 1] ]).T


# In[9]:


print(X)
print(y)


# In[10]:


hist1, model1 = train_single_layer(X, y, lr=1, max_epoch=10)


# In[11]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

def show_graph(history, size=[9,5]):
    plt.rcParams['figure.figsize'] = size
    plt.plot(history)
    plt.xlabel('epoch')
    plt.ylabel('training mse')
    plt.title('Training MSE History')
    plt.show()   


# In[12]:


show_graph(hist1)


# In[13]:


def test_single_layer(X, weights):
    W1, b1 = weights
    # forward pass
    V1 = affine_forward(X, W1, b1)
    y_hat = sigmoid_forward(V1)

    return np.round(y_hat)


# In[14]:


y_pred = test_single_layer(X, model1)
print(y_pred)


# In[15]:


hist1, model1 = train_single_layer(X, y, lr=1, max_epoch=10, weights=model1, history=hist1)


# In[16]:


show_graph(hist1)


# In[17]:


y_pred = test_single_layer(X, model1)
print(y_pred)


# In[48]:


# TWO LAYERS


# In[18]:


def train_two_layers(X, y, n_hidden, lr, max_epoch, weights=None, history=None, print_every=10):

    n_data, n_dim = X.shape
    _, n_out = y.shape

    if weights is None:
        W1 = 2 * np.random.random((n_dim, n_hidden)) - 1
        b1 = np.zeros((1, n_hidden))
        W2 = 2 * np.random.random((n_hidden, n_out)) - 1
        b2 = np.zeros((1, n_out))
        history = []
    else:
        W1, b1, W2, b2 = weights

    for ep in range(max_epoch):
        # forward pass
        V1 = affine_forward(X, W1, b1)
        A1 = sigmoid_forward(V1)
        V2 = affine_forward(A1, W2, b2)
        y_hat = sigmoid_forward(V2)

        # calculate loss
        E = y-y_hat
        mse = np.mean(E**2)
        history.append(mse)

        if ep % print_every == 0:
            acc = np.sum(y==np.round(y_hat))/n_data
            print('epoch: %i/%i, mse: %.7f, acc: %.2f' % (ep, max_epoch, mse, acc))

        # backward pass
        dV2 = sigmoid_backward(E, y_hat)
        dA1, dW2, db2 = affine_backward(dV2, A1, W2, b2)
        dV1 = sigmoid_backward(dA1, A1)
        dX, dW1, db1 = affine_backward(dV1, X, W1, b1)

        # weights update
        W1 = W1 + lr*dW1
        b1 = b1 + lr*db1
        W2 = W2 + lr*dW2
        b2 = b2 + lr*db2
        
    weights = (W1, b1, W2, b2)

    return history, weights


# In[19]:


def test_two_layers(X, weights):
    W1, b1, W2, b2 = weights

    # forward pass
    V1 = affine_forward(X, W1, b1)
    A1 = sigmoid_forward(V1)
    V2 = affine_forward(A1, W2, b2)
    y_hat = sigmoid_forward(V2)
    return np.round(y_hat)


# In[20]:


y = np.array([ [0, 1, 1, 0] ]).T
print(y)


# In[21]:


hist2, model2 = train_two_layers(X, y, n_hidden=4, lr=1, max_epoch=100, print_every=10)


# In[22]:


show_graph(hist2)


# In[23]:


y_pred = test_two_layers(X, model2)
print(y_pred)


# In[24]:


hist2, model2 = train_two_layers(X, y, n_hidden=4, lr=1, max_epoch=300, print_every=10, weights=model2, history=hist2)


# In[25]:


show_graph(hist2)


# In[26]:


y_pred = test_two_layers(X, model2)
print(y_pred)


# In[27]:


hist3, model3 = train_two_layers(X, y, n_hidden=4, lr=0.1, max_epoch=400, print_every=50)


# In[28]:


show_graph(hist3)


# In[29]:


y_pred = test_two_layers(X, model3)
print(y_pred)


# In[49]:


# LEARNING RATE


# In[30]:


hist3, model3 = train_two_layers(X, y, n_hidden=4, lr=0.1, max_epoch=3000, weights=model3, history=hist3, print_every=100)


# In[31]:


show_graph(hist3)


# In[32]:


y_pred = test_two_layers(X, model3)
print(y_pred)


# In[50]:


# BIGGER DATASET


# In[33]:


from sklearn.datasets import make_classification

COLORS = ['red', 'blue']
DIM = 20
INFO = 10
CLASS = 2
NDATA = 600

Xb, yb1 = make_classification(n_samples=NDATA, n_classes=CLASS, n_features=DIM, n_informative=INFO,
                              n_clusters_per_class=4, flip_y=0.2, random_state=33)
yb = yb1.reshape((-1, 1))


# In[34]:


from mpl_toolkits.mplot3d import Axes3D

# shown features
ft = [0, 1, 2]

fig = plt.figure(figsize=(10, 6), dpi=100)
ax = Axes3D(fig)
ax.scatter( Xb[yb1==0, ft[0]], Xb[yb1==0, ft[1]], Xb[yb1==0, ft[2]], c=COLORS[0], marker='s' )							  
ax.scatter( Xb[yb1==1, ft[0]], Xb[yb1==1, ft[1]], Xb[yb1==1, ft[2]], c=COLORS[1], marker='o' )
plt.show()


# In[35]:


hist4, model4 = train_two_layers(Xb, yb, n_hidden=4, lr=1, max_epoch=500, print_every=25)


# In[36]:


show_graph(hist4)


# In[37]:


hist5, model5 = train_two_layers(Xb, yb, n_hidden=4, lr=0.01, max_epoch=500, print_every=25)


# In[38]:


show_graph(hist5)


# In[51]:


# LAYER SIZE


# In[39]:


hist6, model6 = train_two_layers(Xb, yb, n_hidden=20, lr=0.01, max_epoch=500, print_every=25)


# In[40]:


show_graph(hist6)


# In[41]:


def show_n_graph(histories, names, size=[9, 5]):
    plt.rcParams['figure.figsize'] = size
    for i in range(len(histories)):
        plt.plot(histories[i], label=names[i])
    plt.xlabel('epoch')
    plt.ylabel('training mse')
    plt.title('Training MSE History')
    plt.legend()
    plt.show()


# In[42]:


show_n_graph([hist5, hist6], ['4 hidden', '20 hidden'])


# In[52]:


# FULL BATCH GRADIENT DESCENT


# In[53]:


W1, b1, W2, b2 = model6

print('size data train = ', Xb.shape, ': ', Xb.nbytes, 'byte')
print('size weight W1  = ', W1.shape, ': ', W1.nbytes, 'byte')
print('size weight b1  = ', b1.shape, ': ', b1.nbytes, 'byte')
print('size weight W2  = ', W2.shape, ': ', W2.nbytes, 'byte')
print('size weight b2  = ', b2.shape, ': ', b2.nbytes, 'byte')


# In[54]:


V1 = affine_forward(Xb, W1, b1)
A1 = sigmoid_forward(V1)
V2 = affine_forward(A1, W2, b2)
y_hat = sigmoid_forward(V2)

print('size weight A1  = ', A1.shape, ': ', A1.nbytes, 'byte')
print('size weight y_hat  = ', y_hat.shape, ': ', y_hat.nbytes, 'byte')


# In[55]:


E = yb-y_hat

dV2 = sigmoid_backward(E, y_hat)
dA1, dW2, db2 = affine_backward(dV2, A1, W2, b2)
dV1 = sigmoid_backward(dA1, A1)
dX, dW1, db1 = affine_backward(dV1, Xb, W1, b1)

print('size loss matrix = ', E.shape, ': ', E.nbytes, 'byte')
print('size gradient A1 = ', dA1.shape, ': ', dA1.nbytes, 'byte')
print('size gradient W1 = ', dW1.shape, ': ', dW1.nbytes, 'byte')
print('size gradient b1 = ', db1.shape, ': ', db1.nbytes, 'byte')
print('size gradient W2 = ', dW2.shape, ': ', dW2.nbytes, 'byte')
print('size gradient b2 = ', db2.shape, ': ', db2.nbytes, 'byte')


# In[56]:


total_memory = (
    Xb.nbytes + W1.nbytes + W2.nbytes +
    b1.nbytes + b2.nbytes + A1.nbytes +
    y_hat.nbytes + E.nbytes + dA1.nbytes +
    dW1.nbytes + dW2.nbytes + db1.nbytes + db2.nbytes
)

print('total memory for 1 epoch =', total_memory/1000, ' KB')


# In[57]:


# STOCHASTIC GRADIENT DESCENT


# In[58]:


from sklearn.utils import shuffle

def train_two_layers_sgd(X, y, n_hidden, lr, max_epoch, weights=None, history=None, print_every=10):

    n_data, n_dim = X.shape
    _, n_out = y.shape

    if weights is None:
        W1 = 2*np.random.random((n_dim, n_hidden))-1
        b1 = np.zeros((1,n_hidden))
        W2 = 2*np.random.random((n_hidden, n_out))-1
        b2 = np.zeros((1,n_out))
        history = []
    else:
        W1, b1, W2, b2 = weights

    for ep in range(max_epoch):

        X, y = shuffle(X, y)

        for i in range(n_data):
            xs = X[i].reshape(1, -1)
            ys = y[i]

            V1 = affine_forward(xs, W1, b1)
            A1 = sigmoid_forward(V1)
            V2 = affine_forward(A1, W2, b2)
            y_hat = sigmoid_forward(V2)

            E = ys - y_hat

            dV2 = sigmoid_backward(E, y_hat)
            dA1, dW2, db2 = affine_backward(dV2, A1, W2, b2)
            dV1 = sigmoid_backward(dA1, A1)
            dxs, dW1, db1 = affine_backward(dV1, xs, W1, b1)

            W2 = W2 + lr*dW2
            b2 = b2 + lr*db2
            W1 = W1 + lr*dW1
            b1 = b1 + lr*db1

        y_hat = test_two_layers(X, (W1, b1, W2, b2))
        E = y-y_hat
        mse = np.mean(E**2)
        history.append(mse)

        if ep % print_every==0:
            acc = np.sum(y==np.round(y_hat))/n_data
            print('epoch: %i/%i, mse: %.7f, acc: %.2f' % (ep, max_epoch, mse, acc))		

    weights = (W1, b1, W2, b2)
    return history, weights


# In[59]:


hist7, model7 = train_two_layers(Xb, yb, n_hidden=20, lr=0.01, max_epoch=500, print_every=50)


# In[60]:


hist8, model8 = train_two_layers_sgd(Xb, yb, n_hidden=20, lr=0.01, max_epoch=500, print_every=50)


# In[61]:


show_n_graph([hist7, hist8], ['Full-GD', 'SGD'])


# In[62]:


# MINI-BATCH GRADIENT DESCENT


# In[67]:


def train_two_layers_minibatch(X, y, n_hidden, lr, max_epoch, batch, weights=None, history=None, print_every=10):

    n_data, n_dim = X.shape
    _, n_out = y.shape
    n_batch = n_data//batch

    if weights is None:
        W1 = 2*np.random.random((n_dim, n_hidden))-1
        b1 = np.zeros((1,n_hidden))
        W2 = 2*np.random.random((n_hidden, n_out))-1
        b2 = np.zeros((1,n_out))
        history = []
    else:
        W1, b1, W2, b2 = weights

    for ep in range(max_epoch):

        X, y = shuffle(X, y)
        Xb = X.reshape((batch, n_batch, n_dim))
        yb = y.reshape((batch, -1))

        for i in range(Xb.shape[0]):
            xs = Xb[i]
            ys = yb[i].reshape(1, -1).T

            V1 = affine_forward(xs, W1, b1)
            A1 = sigmoid_forward(V1)
            V2 = affine_forward(A1, W2, b2)
            y_hat = sigmoid_forward(V2)

            E = ys - y_hat

            dV2 = sigmoid_backward(E, y_hat)
            dA1, dW2, db2 = affine_backward(dV2, A1, W2, b2)
            dV1 = sigmoid_backward(dA1, A1)
            dxs, dW1, db1 = affine_backward(dV1, xs, W1, b1)

            W2 = W2 + lr*dW2
            b2 = b2 + lr*db2
            W1 = W1 + lr*dW1
            b1 = b1 + lr*db1

        y_hat = test_two_layers(X, (W1, b1, W2, b2))
        E = y-y_hat
        mse = np.mean(E**2)
        history.append(mse)
        
        if ep % print_every==0:
            acc = np.sum(y==np.round(y_hat))/n_data
            print('epoch: %i/%i, mse: %.7f, acc: %.2f' % (ep, max_epoch, mse, acc))		

    weights = (W1, b1, W2, b2)

    return history, weights


# In[68]:


hist9, model9 = train_two_layers_minibatch(Xb, yb, n_hidden=20, lr=0.01, max_epoch=500, batch=10, print_every=50)


# In[69]:


show_n_graph([hist7, hist8, hist9], ['Full-GD', 'SGD', 'Minibatch-GD'])


# In[ ]:




