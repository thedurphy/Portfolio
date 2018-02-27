
# coding: utf-8

# # L-Layers Generalized Neural Network
# 
# ## Table of Contents
# 
# 1. Import/Prep Test Dataset
# 2. Create Activation functions (Forward and Backward)
# 3. Individual Functions
#     - Initializing Parameters
#     - Forward Propogation
#     - Compute Cost
#     - Backward Propogation
#     - Gradients
#     - Update Parameters
# 9. Step-by-Step

# In[8]:


import numpy as np
import matplotlib.pyplot as plt
import h5py


# ## Import/Prep Test Dataset

# In[9]:


def load_data():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


# In[20]:


# train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
# print (train_x_orig.shape)
# print (train_y.shape)
# print (test_x_orig.shape)
# print (test_y.shape)
# print (train_y.shape)
# print (classes.shape)


# # Flatten the training and testing input sets and normalize them

# # In[23]:


# train_x = train_x_orig.reshape(train_x_orig.shape[0],-1).T/255
# test_x = test_x_orig.reshape(test_x_orig.shape[0],-1).T/255
# print (train_x.shape)
# print (test_x.shape)


# # ## Create Activation Functions

# # ### Sigmoid

# # In[65]:


def sigmoid(x):
    return 1/(1+np.exp(-x)), x
# sigmoid(3)

def sigmoid_backward(dA, cache):
    s = 1/(1+np.exp(-cache))
    return dA*s*(1-s)


# ### RELU

# In[148]:


def relu(x):
    return np.maximum(0, x), x
def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z<=0] = 0
    return dZ


# ## Individual Functions

# ### Parameters

# In[191]:


# layer_dims = [12288, 20, 7, 5, 1]
def par_deep(layer_dims):
    np.random.seed(1)
    parameters = {}
    L = len(layer_dims)
    for l in range(1,L):
        parameters.update({
                            'W{0}'.format(l) : np.random.randn(layer_dims[l], layer_dims[l-1])/np.sqrt(layer_dims[l-1]),
                            'b{0}'.format(l) : np.zeros((layer_dims[l], 1))
                          })
    return parameters


# In[39]:


# layer_dims = [12288, 20, 7, 5, 1]
# [l for l in range(1,len(layer_dims))]


# ### Linear Forward Calculations

# In[52]:


def l_forward(A, W, b):
    return W.dot(A)+b, (A,W,b)


# ### Forward Activation

# In[85]:


def lact_forward(A_prev, W, b, activation = 'sigmoid'):
    Z, l_cache = l_forward(A_prev, W, b)
    if activation == 'relu':
        A, a_cache = relu(Z)
        return A, (l_cache, a_cache)
    A, a_cache = sigmoid(Z)
    return A, (l_cache, a_cache)


# ### Forward Propogation

# In[213]:


def L_forward(X, parameters):
    caches = []
    A = X
    L = len(parameters)//2
    for l in range(1, L):
        A_prev = A
        A, cache = lact_forward(A_prev, parameters['W{0}'.format(l)], parameters['b{0}'.format(l)], 'relu')
        caches.append(cache)
    AL, cache = lact_forward(A, parameters["W{0}".format(L)], parameters["b{0}".format(L)], 'sigmoid')
    caches.append(cache)
    return AL, caches


# ### Compute Cost

# In[205]:


def cost(AL, Y):
    return np.squeeze((1./Y.shape[1])*(-np.dot(Y, np.log(AL).T)- np.dot(1-Y, np.log(1-AL).T)))


# ### Linear Backward Calculations

# In[139]:


def l_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = 1./m * np.dot(dZ, A_prev.T)
    db = 1./m * np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db


# ### Backward Activation

# In[67]:


def lact_backward(dA, cache, activation = 'relu'):
    l_cache, a_cache = cache
    if activation == 'sigmoid':
        dZ = sigmoid_backward(dA, a_cache)
    elif activation == 'relu':
        dZ = relu_backward(dA, a_cache)
    dA_prev, dW, db = l_backward(dZ, l_cache)
    return dA_prev, dW, db


# ### Backward Propogation

# In[164]:


def L_backward(AL, Y, caches):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    dAL = - (np.divide(Y,AL) - np.divide(1-Y, 1-AL)) # derivative of the cost function
    current_cache = caches[L-1]
    grads["dA{0}".format(L)], grads['dW{0}'.format(L)], grads['db{0}'.format(L)] = lact_backward(dAL, current_cache, 'sigmoid')
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        grads['dA{0}'.format(l+1)], grads['dW{0}'.format(l+1)], grads['db{0}'.format(l+1)] = lact_backward(grads['dA{0}'.format(l+2)], 
                                                                                                     current_cache)
    return grads


# ### Update Parameters

# In[260]:


def update_par(parameters, grads, learning_rate):
    L = len(parameters)//2
    for l in range(L):
        parameters['W{0}'.format(l+1)] = parameters['W{0}'.format(l+1)] - learning_rate*grads['dW{0}'.format(l+1)]
        parameters['b{0}'.format(l+1)] = parameters['b{0}'.format(l+1)] - learning_rate*grads['db{0}'.format(l+1)]
    return parameters


# ### Prediction

# In[285]:


def predict(X, y, parameters):
    probas, cache = L_forward(X, parameters)
    p = np.vectorize(lambda x:1 if x>0.5 else 0)(probas)
    accuracy = np.sum((p == y)/X.shape[1])
    print("Accuracy: "  + str(accuracy))
    return accuracy, p, probas


# ## Step-by-Step

# In[73]:


# layer_dims


# # ### Parameters Setting

# # In[192]:


# parameters = par_deep(layer_dims)
# print (parameters)


# # ### Linear Outputs of the first Layer
# # $$W^{[1]T}A^{[0]}+b^{[1]}$$
# # *where*
# # $$A^{[0]} = X$$
# # *Output*
# # $$Z^{[1]}$$

# # In[198]:


# l_forward(train_x, parameters['W1'], parameters['b1'])[0]


# # *and*
# # $$A^{[0]}, W^{[1]}, b^{[1]}$$

# # In[114]:


# l_forward(train_x, parameters['W1'], parameters['b1'])[1]


# # ### Linear Activations of the First Layer

# # Linear Output of the first layer, activated with RELU (**RE**ctified **L**inear **U**nit)
# # $$relu(W^{[l]T}A^{[l-1]}b^{[l]})$$
# # **Output**
# # $$A^{[l]}, ((A^{[l-1]}, W^{[l]}, b^{[l]}), Z^{[l]})$$

# # Example: $$A^{[1]}$$

# # In[201]:


# lact_forward(train_x, parameters['W1'], parameters['b1'], 'relu')[0]


# # ### Complete Forward Propogation

# # All L-1 layers are activated using RELU while the final layer(outcome) is activated using sigmoid(binary classification).
# # *Output*
# # $$A^{[l]}$$

# # In[215]:


# AL, cache = L_forward(train_x, parameters)
# len(cache)


# # ### Compute Costs

# # In[216]:


# cost(AL, train_y)


# # ## Derivative outputs of the last layer

# # Each `cache` element houses...
# # $$A^{[l]}, ((A^{[l-1]}, W^{[l]}, b^{[l]}), Z^{[l]})$$
# # *where*
# # 
# # $A^{[l]}$ : Layer $l$'s activated output
# # 
# # $A^{[l-1]}$ : Previous layer's activated output.  $A^{[0]}$ corresponds to the input layer, $X$
# # 
# # $W^{[l]}$ : Corresponding weights for the layer $l$
# # 
# # $b^{[l]}$ : Biases
# # 
# # $Z^{[l]}$ : Linear outputs of layer $l$
# # 
# # Let's print out the $Z^{[L]}$

# # In[217]:


# l, a = cache[-1] # the last layer
# a


# # Since this is the final layer, we cannot derive the $dAL$ from the next layer since there is 
# # For this layer, we used a $\sigma(x)$ activation, so we must use the derivation.

# # In[220]:


# dAL = - (np.divide(train_y, AL)-np.divide(1-train_y, 1-AL))
# dAL


# # The `sigmoid_backward`, or derivative of the sigmoid function, will take a the $dAL$ and the activation_cache, `a`, which is also the $Z^{[L]}$.

# # In[221]:


# sigmoid_backward(dAL, a)


# # The result of sigmoid_backward is then inserted to the linear_backward function with the linear cache which gives you the...
# # $$dA^{[L-1]}, dW^{[L]}, db^{[L]}$$

# # Example : 
# # $$dA^{[L-1]}$$

# # In[222]:


# dA_prev, dW, db = l_backward(sigmoid_backward(dAL, a), l)
# dA_prev


# # ...The gradients for the current layer, $dW^{[L]}, db^{[L]}$, and the derivative of the activation for the previous layer, $dA^{[L-1]}$, which is used in conjunction with the activation_cache for that layer then inserted in to the proper backward_activation function, which in this case, all $L-1$ (in between) layers are activated with RELU.  The result is then $dZ^{[L-1]}$, and the cycle continues until we reach the first layer. 
# # 
# # **Note** that this entire process happens nested inside the linear_activation_backward functions which performs the specific backward_activation calculation which gets put inside the linear_backward function.  That function is called the `lact_backward`.  Below, the equivalency and convenience is shown printing the $dW^{[L]}$ using both methods.

# # In[223]:


# print(lact_backward(dAL, cache[-1], 'sigmoid')[1])
# print(l_backward(sigmoid_backward(dAL, a), l)[1])


# # Now we do the full backward propogation to all the layers.  This results in a dictionary of gradients.  Our `layer_dims` was `[12288, 20, 7, 5, 1]`, so we have 3 hidden-layers and 1 output-layer and 1 input-layer.  We should have the following gradients for the hidden layers.
# # \begin{bmatrix}
# # dA^{[3]} & dA^{[2]} & dA^{[1]} \\
# # dW^{[3]} & dW^{[2]} & dW^{[1]} \\
# # db^{[3]} & db^{[2]} & db^{[1]}
# # \end{bmatrix}
# # 
# # **Note** $dA^{[3]}$ is equivalent to the last layer, $dA^{[L]}$

# # In[224]:


# g = L_backward(AL, train_y, cache)
# g.keys()


# # Example : 
# # $$ dA^{[3]} $$

# # In[225]:


# g['dA3']


# # Now we use the `update_parameters` function to iteratively update our old parameters.

# # In[232]:


# print ("Old W1 was")
# print (parameters['W1'])
# print ("Now it is...")
# print(update_par(parameters, g, 0.01)['W1'])


# In[286]:


def L_model(X, Y, layer_dims, learning_rate = 0.0075, iterations = 3000, print_cost=False):
    np.random.seed(1)
    costs = []
    ac = []
    parameters = par_deep(layer_dims)
    for i in range(iterations):
        AL, caches = L_forward(X, parameters)
        c = cost(AL, Y)
        grads = L_backward(AL, Y, caches)
        parameters = update_par(parameters, grads, learning_rate)
        if print_cost and i%100==0:
            print ("Cost after iteration {0}: {1}".format(i, c))
            costs.append(c)
            accuracy = predict(X, Y, parameters)[0]
            ac.append(accuracy)
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    plt.plot(ac)
    plt.ylabel('Accuracy Score')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    return parameters


# In[292]:


# p = L_model(train_x, train_y, [12288,25, 20, 15, 10, 5, 1], iterations = 3000, print_cost = True)
# predict(test_x, test_y, p)


# In[247]:


def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):#lr was 0.009
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
    
    Arguments:
    X -- data, numpy array of shape (number of examples, num_px * num_px * 3)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    np.random.seed(1)
    costs = []                         # keep track of cost
    
    # Parameters initialization.
    ### START CODE HERE ###
    parameters = par_deep(layers_dims)
    ### END CODE HERE ###
    
    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        ### START CODE HERE ### (≈ 1 line of code)
        AL, caches = L_forward(X, parameters)
        ### END CODE HERE ###
        
        # Compute cost.
        ### START CODE HERE ### (≈ 1 line of code)
        c = cost(AL, Y)
        ### END CODE HERE ###
    
        # Backward propagation.
        ### START CODE HERE ### (≈ 1 line of code)
        grads = L_backward(AL,Y,caches)
        ### END CODE HERE ###
 
        # Update parameters.
        ### START CODE HERE ### (≈ 1 line of code)
        parameters = update_parameters(parameters, grads, learning_rate)
        ### END CODE HERE ###
                
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, c))
        if print_cost and i % 100 == 0:
            costs.append(c)
            
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters

# parameters = L_layer_model(train_x, train_y, layer_dims, num_iterations = 2500, print_cost = True)


# # In[ ]:


# p = predict(train_x, train_y, parameters)
# p = predict(test_x, test_y, parameters)

