# NeuroChan
Simple neural-library which let you create simple fully-connected neural networks.

# Using
Using is really simple, you just have to create instance of Net and add some layers to it. The first argument is count of neurons in a layer you added and the second is activation function.



```ts
let net: Net = new Net();
net.AddLayer(new Layer(784, "none"));
net.AddLayer(new Layer(128, "LeakyReLU"));
net.AddLayer(new Layer(10, "sigmoid"));
```

# Activation Functions
