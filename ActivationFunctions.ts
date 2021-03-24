import { access } from 'fs';
import { Dictionary } from './Dictionary' 

let ActivationFunctions = new Dictionary<string, any>();

function relu(x: number): number { if(x > 0) return x; return x*0.000001; }
function sigmoid(x: number): number { return 1/(1+Math.E**-x); }
function tanhi(x: number): number { return (Math.E**x - Math.E**-x)/(Math.E**x + Math.E**-x); }
function leakyrelu(x: number): number { if( x > 0) return x; return x*0.01; }
function softplus(x: number):number { return Math.log(1+Math.E**x) }
function softsign(x: number):number { return x / (1 + Math.abs(x)) }
function arctg(x: number): number { return Math.atan(x) }
function selu(x: number): number { if(x > 0) return x*1.0507009873554804934193349852946; return 1.0507009873554804934193349852946*(1.6732632423543772848170429916717*Math.E**x-1.6732632423543772848170429916717) }

function __relu(x: number): number { if(x > 0) return 1; return 0; } //works
function __sigmoid(fx: number): number { return fx*(1-fx); } //works
function __tanhi(fx: number): number { return 1-fx**2 } //works
function __leakyrelu(x: number): number { if( x > 0) return 1; return 0.01; } //works
function __softplus(fx: number):number { return (Math.E**fx-1)/(Math.E**fx) } //works
function __softsign(x: number):number { return 1 / (1 + Math.abs(x))**2 }
function __arctg(fx: number): number { return 1/(1+Math.tan(fx)**2) } //works
function __selu(x: number): number { if(x > 0) return 1; return 1.6732632423543772848170429916717*Math.E**x }


ActivationFunctions.Add("relu", [relu, __relu]);
ActivationFunctions.Add("sigmoid", [sigmoid, __sigmoid]);
ActivationFunctions.Add("tanhi", [tanhi, __tanhi]);
ActivationFunctions.Add("leakyrelu", [leakyrelu, __leakyrelu]);
ActivationFunctions.Add("softplus", [softplus, __softplus]);
ActivationFunctions.Add("softsign", [softsign, __softsign]);
ActivationFunctions.Add("arctg", [arctg, __arctg]);
ActivationFunctions.Add("selu", [selu, __selu]);

ActivationFunctions.Add("All", [relu, sigmoid, selu, tanhi, leakyrelu, softplus, softsign, arctg, selu]);
ActivationFunctions.Add("none", [function() {}, function() {}]);

export default ActivationFunctions;