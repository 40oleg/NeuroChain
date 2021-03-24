import { Neuron } from './Neuron' 
import { Dictionary } from './Dictionary' 
import ActivationFunctions from './ActivationFunctions'

export class Layer {
    public neurons: Array<Neuron>;
    public bias: Neuron;
    public func: any;
    public dropout_probability: number;
    constructor(count: number, func: string, probability: number) {
        this.neurons = new Array<Neuron>();
        for(let i = 0; i < count; i++) {
            this.neurons.push(new Neuron());
        }
        this.bias = new Neuron();
        this.func = ActivationFunctions.Key(func);
        this.dropout_probability = probability; 
    }
}