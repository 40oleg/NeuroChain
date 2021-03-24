import { Net } from './Net'
import { Layer } from './Layer'
const fs = require('fs')

// import { data } from './dataset'

let train: any = JSON.parse(fs.readFileSync('./datasets/mnist_handwritten_train.json', 'utf-8'));
let test: any = JSON.parse(fs.readFileSync('./datasets/mnist_handwritten_test.json', 'utf-8'));

let net: Net = new Net();

train = net.Normalize(train, 0, 255);
test = net.Normalize(test, 0, 255);

net.AddLayer(new Layer(784, "none", 0));
net.AddLayer(new Layer(64, "tanhi", 0));
net.AddLayer(new Layer(10, "tanhi", 0));

net.SetTrainSample(train);
net.SetTestSample(test);

net.Train(1000, 0.1, 0.001);

console.log(net.layers[1].neurons)
let count = 0;
let max = 100;
for(let i = 0; i < max; i++) {
    // console.log(Max(net.Run([train[i][0]])[0]), Max(train[i][1]))
    if(Max(net.Run([train[i][0]])[0]) == Max(train[i][1])) {
        count++;
    }
}
console.log("Result Train: "+(count/max))
count = 0;
for(let i = 0; i < max; i++) {
    if(Max(net.Run([test[i][0]])[0]) == Max(test[i][1])) {
        count++;
    }
}
console.log("Result Test: "+(count/max))

function Max(arr: any) {
    let max = 0;
    let val = 0;
    for(let i = 0; i < arr.length; i++) {
        if(arr[i] > val) {
            max = i;
            val = arr[i];
        }
    }
    return max;
}