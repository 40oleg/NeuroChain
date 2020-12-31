# NeuroChan
Simple neural-library which let you create simple fully-connected neural networks.

# Using
Using is really simple, you just have to create instance of Net and add some layers to it.
```js
let net: Net = new Net();
net.AddLayer(new Layer(784, "none", 0));
net.AddLayer(new Layer(128, "LeakyReLU", 0));
net.AddLayer(new Layer(10, "sigmoid", 0));
```

# Activation Functions
