import { Layer } from './Layer' 
import { FunctionList, GPU, KernelOutput } from 'gpu.js';
import ActivationFunctions from './ActivationFunctions'

function relu(x: number): number { if(x > 0) return x; return x*0.000001; } //1
function sigmoid(x: number): number { return 1/(1+Math.E**-x); } //2
function tanhi(x: number): number { return (Math.E**x - Math.E**-x)/(Math.E**x + Math.E**-x); } //3
function leakyrelu(x: number): number { if( x > 0) return x; return x*0.01; } //4
function softplus(x: number):number { return Math.log(1+Math.E**x) } //5
function softsign(x: number):number { return x / (1 + Math.abs(x)) } //6
function arctg(x: number): number { return Math.atan(x) } //7
function selu(x: number): number { if(x > 0) return x*1.0507009873554804934193349852946; return 1.0507009873554804934193349852946*(1.6732632423543772848170429916717*Math.E**x-1.6732632423543772848170429916717) } //7



export class Net {
    public layers: Array<Layer> = new Array<Layer>();
    private trainSample: any;
    private testSample: any;
    public gpu = new GPU();
    constructor() {}
    public multiply_GPU = this.gpu.createKernel(function(a: Array<Array<number>>, b: Array<Array<number>>, w:number, func: number) {
        let sum = 0;
        for (let i = 0; i < w; i++) {
            sum += a[this.thread.y][i] * b[i][this.thread.x];
        }
        switch(func) {
            case 1: return relu(sum);
            case 2: return sigmoid(sum);
            case 3: return tanhi(sum);
            case 4: return leakyrelu(sum);
            case 5: return softplus(sum);
            case 6: return softsign(sum);
            case 7: return arctg(sum);
            case 8: return selu(sum);
            // default: return sum;
        }}).setDynamicOutput(true).setDynamicArguments(true).setFunctions(ActivationFunctions.Key("All"));
    public AddLayer(_layer: Layer): void {
        this.layers.push(_layer);
        if(this.layers.length > 1) this.AddWeights();
    }
    private AddWeights(): void {
        for(let i = 0; i < this.layers[this.layers.length-2].neurons.length; i++) {
            for(let j = 0; j < this.layers[this.layers.length-1].neurons.length; j++) {
                this.layers[this.layers.length-2].neurons[i].connections[j] = this.GetRandomWeight();
            }
        }
        for(let i = 0; i < this.layers[this.layers.length-1].neurons.length; i++) {
            this.layers[this.layers.length-2].bias.connections[i] = this.GetRandomWeight();
        }
        
    }
    private GetRandomWeight(): number {
        return Math.random()-0.5;
        // return Math.floor(Math.random() * 3);
    }
    public Run(arr: number[][]): number[][] {
        // this.SetInitValue(arr);
        let start;
        let batch = arr.slice();
        let weights: Array<Array<number>>;
        for(let i = 0; i < this.layers.length-1; i++) { 
            weights = [];
            for(let j = 0; j < this.layers[i].neurons.length; j++) {
                weights.push([]);
                for(let k = 0; k < this.layers[i+1].neurons.length; k++) {
                    weights[j][k] = this.layers[i].neurons[j].connections[k];
                }
            }
            start = Date.now();
            this.multiply_GPU = this.multiply_GPU.setOutput([weights[0].length, batch.length]);
            batch = this.multiply_GPU(batch, weights, weights.length, ActivationFunctions.GetIndex(this.layers[i+1].func[0].name)) as number[][];
            // console.log("Mul: "+(Date.now()-start));
            for(let j = 0; j < batch.length; j++) {
                for(let k = 0; k < batch[0].length; k++) {
                    this.layers[i+1].neurons[k].activatedValue[j] = batch[j][k];
                }
            }
        }
        return batch as number[][];
    }
    private SetInitValue(arr: number[][]): void {
        for(let i = 0; i < arr.length; i++) {
            for(let j = 0; j < arr.length; j++) {
                this.layers[0].neurons[i].activatedValue[j] = arr[i][j];
            }
        }
    }
    public SetTrainSample(arr: any) {
        this.trainSample = arr;
    }
    public SetTestSample(arr: any) {
        this.testSample = arr;
    }
    public Train(iters: number, learning_rate_start: number, learning_rate_end: number): void {
        for(let i = 0; i < iters; i++) {
            let learning_rate = learning_rate_start - (learning_rate_start - learning_rate_end)/iters*i;
            let errors: number[][] = this.Round();
            if(i % (iters/10) == 0) {
                console.log('\x1Bc');
                console.log("Progress:"+Math.round((i+1)/iters*100)+"%");
            }
            this.BackPropogation(errors, learning_rate);
        }
    }
    private BackPropogation(errors: number[][], learning_rate: number): void {
        for(let i = this.layers.length; i > 1; i--) {
            if(i == this.layers.length) {
                for(let j = 0; j < this.layers[i-1].neurons.length; j++) {
                    this.layers[i-1].neurons[j].grad = 0;
                    for(let k = 0; k < errors[j].length; k++) {
                        let x = this.layers[i-1].neurons[j].activatedValue[k];
                        this.layers[i-1].neurons[j].grad += errors[j][k] * this.layers[i-1].func[1](x);
                    }
                }
            } else {
                for(let j = 0; j < this.layers[i-1].neurons.length; j++) {
                    let sum = 0;
                    for(let k = 0; k < this.layers[i].neurons.length; k++) {
                        sum += this.layers[i].neurons[k].grad * this.layers[i-1].neurons[j].connections[k];
                    }
                    let grad = 0;
                    for(let k = 0; k < this.layers[i-1].neurons[j].activatedValue.length; k++) {
                        let x = this.layers[i-1].neurons[j].activatedValue[k];
                        grad += sum * this.layers[i-1].func[1](x);
                    }
                    this.layers[i-1].neurons[j].grad = grad;
                }
            }
            //может в весах дело
            for(let j = 0; j < this.layers[i-2].neurons.length; j++) {
                for(let k = 0; k < this.layers[i-1].neurons.length; k++) {
                        this.layers[i-2].neurons[j].connections[k] += (this.layers[i-2].neurons[j].connections[k]*0.9999999) - learning_rate * this.layers[i-1].neurons[k].grad * this.layers[i-2].neurons[j].activatedValue[q];
                    
                }
            }
            for(let j = 0; j < this.layers[i-1].neurons.length; j++) {
                this.layers[i-2].bias.connections[j] -= learning_rate * this.layers[i-1].neurons[j].grad;
            }
        }
    }
    public Normalize(_data: any, min: number, max: number): any {
        for(let i = 0; i < _data.length; i++) {
            for(let j = 0; j < _data[i][0].length; j++) {
                _data[i][0][j] = (_data[i][0][j]-min)/(max-min);
            }
        }
        return _data;
    }
    // public Test(repetitions: number): object {
    //     let count = 0;
    //     let max = repetitions;
    //     let result = { train: 0,test: 0 };
    //     for(let i = 0; i < max; i++) {
    //         if(Max(this.Run(this.trainSample[i][0])) == Max(this.trainSample[i][1])) {
    //             count++;
    //         }
    //     }
    //     result.train = count/max;
    //     count = 0;
    //     for(let i = 0; i < max; i++) {
    //         if(Max(this.Run(this.testSample[i][0])) == Max(this.testSample[i][1])) {
    //             count++;
    //         }
    //     }
    //     result.test = count/max;
    //     return result;
    // }
    private Round(): number[][] {
        let input = [];
        let rightAnswer = [];
        for(let i = 0; i < 100; i++) {
            let randomItem = this.trainSample[Math.floor(Math.random()*this.trainSample.length)];
            input.push(randomItem[0]);
            rightAnswer.push(randomItem[1]);
        }
        let netOutput = this.Run(input);
        let errors: number[][] = [];
        for(let j = 0; j < netOutput.length; j++) {
            errors.push([]);
            for(let i = 0; i < netOutput[j].length; i++) {
                errors[j][i] = ((netOutput[j][i] - rightAnswer[j][i])**2);
            }
        }
        // console.log(errors)
        return errors;
    }
}

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


