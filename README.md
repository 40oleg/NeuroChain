# NeuroChan.js
Simple neural-library which let you create simple fully-connected neural networks.

# Using
Using is really simple, you just have to create instance of Net and add some layers to it. The first argument is count of neurons in a layer you added and the second is an activation function. The first layer doesn't have any activation function because it isn't necessary. You have to state this explicity.



```ts
let net: Net = new Net();
net.AddLayer(new Layer(10, "none"));
net.AddLayer(new Layer(12, "LeakyReLU"));
net.AddLayer(new Layer(2, "sigmoid"));
```

# Activation Functions
There are several activation functions at this moment.
* sigmoid
* ReLU
* LeakyReLU
* tanh
