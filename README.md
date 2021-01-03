# NeuroChan.js
Simple neural-library which let you create simple fully-connected neural networks. Wrote on TypeScript.

# Using
You just have to create instance of Net and add some layers to it. The first argument is count of neurons in a layer you added and the second is an activation function. The first layer doesn't have any activation function because it isn't necessary, but you have to state this explicity.
Next you train net and can check quality, but now you can't save weights, therefore you have to learn your net every time you want to use it.


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
* Leaky ReLU
* hyperbolic tangent
* softplus
* softsign
* argtg

# Example of simplest neural network
```ts
//dataset
let train = [
  [[0,0],[0]],
  [[0,1],[1]],
  [[1,0],[1]],
  [[1,1],[0]]
];

//making instance of dataset
let net: Net = new Net();

//adding layers
net.AddLayer(new Layer(2, "none", 0));
net.AddLayer(new Layer(3, "tanh", 0));
net.AddLayer(new Layer(1, "sigmoid", 0));

//setting train and test sample
net.SetTrainSample(train);

//training net
net.Train(100000, 0.1, 0.01);

//check result
console.log(net.Run([0,0])); //0.0468798408364958
console.log(net.Run([0,1])); //0.9355276250031281
console.log(net.Run([1,0])); //0.9339797152023931
console.log(net.Run([1,1])); //0.0675244824863299
```

# Performance
You should use NeuroChan only educational purposes because of speed of learning and lack of usefull technologies.

# Methods
```js
net.Normalize(sample, min_value, max_value); // normalization input values
// sample - your dataset in the standart format (3D-array)
// min_value - minimal value in the dataset
// max_value - maximal value in the dataset

net.Augmentaion(sample, coef); //sample augmentaion (DOESNT'T SUPPORT)
// sample - your dataset in the standart format (3D-array)
// coef - augmentaion coefficient

net.Train(count_of_repetitions, starting_learing_rate, ending_learning_rate);
// count_of_repetitions - value which have to be more then train-sample 3-5 times

net.Test(repetitions); // evaluation method which prints quality of learning
// before using this method you have to set test sample with method .SetTestSample(test_sample)
// repetitions - count of examples used to define quality


```
