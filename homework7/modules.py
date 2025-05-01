import numpy as np
import scipy as sp
import scipy.signal

class Module(object):
    def __init__(self):
        self.output = None
        self.gradInput = None
        self.training = True
    
    def forward(self, input):
        return self.updateOutput(input)

    def backward(self, input, gradOutput):
        self.updateGradInput(input, gradOutput)
        self.accGradParameters(input, gradOutput)
        return self.gradInput
    
    def updateOutput(self, input):
        pass

    def updateGradInput(self, input, gradOutput):
        pass   
    
    def accGradParameters(self, input, gradOutput):
        pass
    
    def zeroGradParameters(self): 
        pass
        
    def getParameters(self):
        return []
        
    def getGradParameters(self):
        return []
    
    def train(self):
        self.training = True
    
    def evaluate(self):
        self.training = False
    
    def __repr__(self):
        return "Module"

class Sequential(Module):
    def __init__(self):
        super(Sequential, self).__init__()
        self.modules = []
   
    def add(self, module):
        self.modules.append(module)

    def updateOutput(self, input):
        self.output = input
        for module in self.modules:
            self.output = module.forward(self.output)
        return self.output

    def backward(self, input, gradOutput):
        self.gradInput = gradOutput
        for i in range(len(self.modules)-1, 0, -1):
            self.gradInput = self.modules[i].backward(self.modules[i-1].output, self.gradInput)
        self.gradInput = self.modules[0].backward(input, self.gradInput)
        return self.gradInput
      
    def zeroGradParameters(self): 
        for module in self.modules:
            module.zeroGradParameters()
    
    def getParameters(self):
        return [x.getParameters() for x in self.modules]
    
    def getGradParameters(self):
        return [x.getGradParameters() for x in self.modules]
    
    def __repr__(self):
        string = "".join([str(x) + '\n' for x in self.modules])
        return string
    
    def __getitem__(self,x):
        return self.modules.__getitem__(x)
    
    def train(self):
        self.training = True
        for module in self.modules:
            module.train()
    
    def evaluate(self):
        self.training = False
        for module in self.modules:
            module.evaluate()

class Linear(Module):
    def __init__(self, n_in, n_out):
        super(Linear, self).__init__()
       
        stdv = 1./np.sqrt(n_in)
        self.W = np.random.uniform(-stdv, stdv, size = (n_out, n_in))
        self.b = np.random.uniform(-stdv, stdv, size = n_out)
        
        self.gradW = np.zeros_like(self.W)
        self.gradb = np.zeros_like(self.b)
        
    def updateOutput(self, input):
        self.output = input.dot(self.W.T) + self.b
        return self.output
    
    def updateGradInput(self, input, gradOutput):
        self.gradInput = gradOutput.dot(self.W)
        return self.gradInput
    
    def accGradParameters(self, input, gradOutput):
        self.gradW = gradOutput.T.dot(input)
        self.gradb = gradOutput.sum(axis=0)
    
    def zeroGradParameters(self):
        self.gradW.fill(0)
        self.gradb.fill(0)
        
    def getParameters(self):
        return [self.W, self.b]
    
    def getGradParameters(self):
        return [self.gradW, self.gradb]
    
    def __repr__(self):
        s = self.W.shape
        q = 'Linear %d -> %d' %(s[1],s[0])
        return q

class SoftMax(Module):
    def __init__(self):
         super(SoftMax, self).__init__()
    
    def updateOutput(self, input):
        input_exp = np.exp(input - input.max(axis=1, keepdims=True))
        self.output = input_exp / input_exp.sum(axis=1, keepdims=True)
        return self.output
    
    def updateGradInput(self, input, gradOutput):
        grad = self.output * gradOutput
        self.gradInput = grad - self.output * grad.sum(axis=1, keepdims=True)
        return self.gradInput
    
    def __repr__(self):
        return "SoftMax"

class LogSoftMax(Module):
    def __init__(self):
         super(LogSoftMax, self).__init__()
    
    def updateOutput(self, input):
        input = input - input.max(axis=1, keepdims=True)
        exp_input = np.exp(input)
        sum_exp = exp_input.sum(axis=1, keepdims=True)
        self.output = input - np.log(sum_exp)
        return self.output
    
    def updateGradInput(self, input, gradOutput):
        exp_input = np.exp(input - input.max(axis=1, keepdims=True))
        sum_exp = exp_input.sum(axis=1, keepdims=True)
        self.gradInput = gradOutput - exp_input / sum_exp * gradOutput.sum(axis=1, keepdims=True)
        return self.gradInput
    
    def __repr__(self):
        return "LogSoftMax"

class BatchNormalization(Module):
    EPS = 1e-3
    def __init__(self, alpha = 0.):
        super(BatchNormalization, self).__init__()
        self.alpha = alpha
        self.moving_mean = None 
        self.moving_variance = None
        
    def updateOutput(self, input):
        if self.training:
            mean = input.mean(axis=0)
            var = input.var(axis=0) + self.EPS
            
            if self.moving_mean is None:
                self.moving_mean = mean
                self.moving_variance = var
            else:
                self.moving_mean = self.alpha * self.moving_mean + (1 - self.alpha) * mean
                self.moving_variance = self.alpha * self.moving_variance + (1 - self.alpha) * var
            
            self.output = (input - mean) / np.sqrt(var)
        else:
            self.output = (input - self.moving_mean) / np.sqrt(self.moving_variance)
        return self.output
    
    def updateGradInput(self, input, gradOutput):
        if self.training:
            mean = input.mean(axis=0)
            var = input.var(axis=0) + self.EPS
            std = np.sqrt(var)
            
            n = input.shape[0]
            x_centered = input - mean
            x_normalized = x_centered / std
            
            dvar = np.sum(gradOutput * x_centered * (-0.5) * (var ** (-1.5)), axis=0)
            dmean = np.sum(gradOutput * (-1 / std), axis=0) + dvar * np.mean(-2 * x_centered, axis=0)
            
            self.gradInput = gradOutput / std + dvar * 2 * x_centered / n + dmean / n
        else:
            self.gradInput = gradOutput / np.sqrt(self.moving_variance)
        return self.gradInput
    
    def __repr__(self):
        return "BatchNormalization"

class ChannelwiseScaling(Module):
    def __init__(self, n_out):
        super(ChannelwiseScaling, self).__init__()

        stdv = 1./np.sqrt(n_out)
        self.gamma = np.random.uniform(-stdv, stdv, size=n_out)
        self.beta = np.random.uniform(-stdv, stdv, size=n_out)
        
        self.gradGamma = np.zeros_like(self.gamma)
        self.gradBeta = np.zeros_like(self.beta)

    def updateOutput(self, input):
        self.output = input * self.gamma + self.beta
        return self.output
        
    def updateGradInput(self, input, gradOutput):
        self.gradInput = gradOutput * self.gamma
        return self.gradInput
    
    def accGradParameters(self, input, gradOutput):
        self.gradBeta = np.sum(gradOutput, axis=0)
        self.gradGamma = np.sum(gradOutput*input, axis=0)
    
    def zeroGradParameters(self):
        self.gradGamma.fill(0)
        self.gradBeta.fill(0)
        
    def getParameters(self):
        return [self.gamma, self.beta]
    
    def getGradParameters(self):
        return [self.gradGamma, self.gradBeta]
    
    def __repr__(self):
        return "ChannelwiseScaling"

class Dropout(Module):
    def __init__(self, p=0.5):
        super(Dropout, self).__init__()
        
        self.p = p
        self.mask = None
        
    def updateOutput(self, input):
        if self.training:
            self.mask = (np.random.random(input.shape) > self.p) / (1 - self.p)
            self.output = input * self.mask
        else:
            self.output = input
        return self.output
    
    def updateGradInput(self, input, gradOutput):
        if self.training:
            self.gradInput = gradOutput * self.mask
        else:
            self.gradInput = gradOutput
        return self.gradInput
        
    def __repr__(self):
        return "Dropout"

class ReLU(Module):
    def __init__(self):
         super(ReLU, self).__init__()
    
    def updateOutput(self, input):
        self.output = np.maximum(input, 0)
        return self.output
    
    def updateGradInput(self, input, gradOutput):
        self.gradInput = np.multiply(gradOutput , input > 0)
        return self.gradInput
    
    def __repr__(self):
        return "ReLU"

class LeakyReLU(Module):
    def __init__(self, slope = 0.03):
        super(LeakyReLU, self).__init__()
            
        self.slope = slope
        
    def updateOutput(self, input):
        self.output = np.where(input > 0, input, input * self.slope)
        return self.output
    
    def updateGradInput(self, input, gradOutput):
        self.gradInput = np.where(input > 0, gradOutput, gradOutput * self.slope)
        return self.gradInput
    
    def __repr__(self):
        return "LeakyReLU"

class ELU(Module):
    def __init__(self, alpha = 1.0):
        super(ELU, self).__init__()
        
        self.alpha = alpha
        
    def updateOutput(self, input):
        self.output = np.where(input > 0, input, self.alpha * (np.exp(input) - 1))
        return self.output
    
    def updateGradInput(self, input, gradOutput):
        self.gradInput = np.where(input > 0, gradOutput, gradOutput * self.alpha * np.exp(input))
        return self.gradInput
    
    def __repr__(self):
        return "ELU"

class SoftPlus(Module):
    def __init__(self):
        super(SoftPlus, self).__init__()
    
    def updateOutput(self, input):
        self.output = np.log(1 + np.exp(input))
        return self.output
    
    def updateGradInput(self, input, gradOutput):
        self.gradInput = gradOutput * (1 - 1 / (1 + np.exp(input)))
        return self.gradInput
    
    def __repr__(self):
        return "SoftPlus"

class Criterion(object):
    def __init__(self):
        self.output = None
        self.gradInput = None
        
    def forward(self, input, target):
        return self.updateOutput(input, target)

    def backward(self, input, target):
        return self.updateGradInput(input, target)
    
    def updateOutput(self, input, target):
        return self.output

    def updateGradInput(self, input, target):
        return self.gradInput   

    def __repr__(self):
        return "Criterion"

class MSECriterion(Criterion):
    def __init__(self):
        super(MSECriterion, self).__init__()
        
    def updateOutput(self, input, target):   
        self.output = np.sum(np.power(input - target,2)) / input.shape[0]
        return self.output 
 
    def updateGradInput(self, input, target):
        self.gradInput  = (input - target) * 2 / input.shape[0]
        return self.gradInput

    def __repr__(self):
        return "MSECriterion"

class ClassNLLCriterionUnstable(Criterion):
    EPS = 1e-15
    def __init__(self):
        super(ClassNLLCriterionUnstable, self).__init__()
        
    def updateOutput(self, input, target): 
        input_clamp = np.clip(input, self.EPS, 1 - self.EPS)
        self.output = -np.sum(target * np.log(input_clamp)) / input.shape[0]
        return self.output

    def updateGradInput(self, input, target):
        input_clamp = np.clip(input, self.EPS, 1 - self.EPS)
        self.gradInput = -target / input_clamp / input.shape[0]
        return self.gradInput
    
    def __repr__(self):
        return "ClassNLLCriterionUnstable"

class ClassNLLCriterion(Criterion):
    def __init__(self):
        super(ClassNLLCriterion, self).__init__()
        
    def updateOutput(self, input, target): 
        self.output = -np.sum(target * input) / input.shape[0]
        return self.output

    def updateGradInput(self, input, target):
        self.gradInput = -target / input.shape[0]
        return self.gradInput
    
    def __repr__(self):
        return "ClassNLLCriterion"

def sgd_momentum(variables, gradients, config, state):  
    state.setdefault('accumulated_grads', {})
    
    var_index = 0 
    for current_layer_vars, current_layer_grads in zip(variables, gradients): 
        for current_var, current_grad in zip(current_layer_vars, current_layer_grads):
            
            old_grad = state['accumulated_grads'].setdefault(var_index, np.zeros_like(current_grad))
            
            np.add(config['momentum'] * old_grad, config['learning_rate'] * current_grad, out=old_grad)
            
            current_var -= old_grad
            var_index += 1     

def adam_optimizer(variables, gradients, config, state):  
    state.setdefault('m', {})  # first moment vars
    state.setdefault('v', {})  # second moment vars
    state.setdefault('t', 0)   # timestamp
    state['t'] += 1
    for k in ['learning_rate', 'beta1', 'beta2', 'epsilon']:
        assert k in config, config.keys()
    
    var_index = 0 
    lr_t = config['learning_rate'] * np.sqrt(1 - config['beta2']**state['t']) / (1 - config['beta1']**state['t'])
    for current_layer_vars, current_layer_grads in zip(variables, gradients): 
        for current_var, current_grad in zip(current_layer_vars, current_layer_grads):
            var_first_moment = state['m'].setdefault(var_index, np.zeros_like(current_grad))
            var_second_moment = state['v'].setdefault(var_index, np.zeros_like(current_grad))
            
            np.add(config['beta1'] * var_first_moment, (1 - config['beta1']) * current_grad, out=var_first_moment)
            np.add(config['beta2'] * var_second_moment, (1 - config['beta2']) * current_grad**2, out=var_second_moment)
            
            current_var -= lr_t * var_first_moment / (np.sqrt(var_second_moment) + config['epsilon'])
            
            assert var_first_moment is state['m'].get(var_index)
            assert var_second_moment is state['v'].get(var_index)
            var_index += 1

class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Conv2d, self).__init__()
        assert kernel_size % 2 == 1, kernel_size
       
        stdv = 1./np.sqrt(in_channels)
        self.W = np.random.uniform(-stdv, stdv, size = (out_channels, in_channels, kernel_size, kernel_size))
        self.b = np.random.uniform(-stdv, stdv, size=(out_channels,))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        self.gradW = np.zeros_like(self.W)
        self.gradb = np.zeros_like(self.b)
        
    def updateOutput(self, input):
        pad_size = self.kernel_size // 2
        padded_input = np.pad(input, ((0,0), (0,0), (pad_size,pad_size), (pad_size,pad_size)), mode='constant')
        
        batch_size = input.shape[0]
        h, w = input.shape[2], input.shape[3]
        output = np.zeros((batch_size, self.out_channels, h, w))
        
        for i in range(batch_size):
            for j in range(self.out_channels):
                for k in range(self.in_channels):
                    output[i,j] += sp.signal.correlate(padded_input[i,k], self.W[j,k], mode='valid')
                output[i,j] += self.b[j]
        
        self.output = output
        return self.output
    
    def updateGradInput(self, input, gradOutput):
        pad_size = self.kernel_size // 2
        padded_grad = np.pad(gradOutput, ((0,0), (0,0), (pad_size,pad_size), (pad_size,pad_size)), mode='constant')
        
        batch_size = input.shape[0]
        h, w = input.shape[2], input.shape[3]
        grad_input = np.zeros_like(input)
        
        for i in range(batch_size):
            for j in range(self.in_channels):
                for k in range(self.out_channels):
                    grad_input[i,j] += sp.signal.correlate(padded_grad[i,k], self.W[k,j,::-1,::-1], mode='valid')
        
        self.gradInput = grad_input
        return self.gradInput
    
    def accGradParameters(self, input, gradOutput):
        pad_size = self.kernel_size // 2
        padded_input = np.pad(input, ((0,0), (0,0), (pad_size,pad_size), (pad_size,pad_size)), mode='constant')
        
        batch_size = input.shape[0]
        
        for i in range(batch_size):
            for j in range(self.out_channels):
                for k in range(self.in_channels):
                    self.gradW[j,k] += sp.signal.correlate(padded_input[i,k], gradOutput[i,j], mode='valid')
        
        self.gradb = gradOutput.sum(axis=(0,2,3))
    
    def zeroGradParameters(self):
        self.gradW.fill(0)
        self.gradb.fill(0)
        
    def getParameters(self):
        return [self.W, self.b]
    
    def getGradParameters(self):
        return [self.gradW, self.gradb]
    
    def __repr__(self):
        s = self.W.shape
        q = 'Conv2d %d -> %d' %(s[1],s[0])
        return q

class MaxPool2d(Module):
    def __init__(self, kernel_size):
        super(MaxPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.gradInput = None
                    
    def updateOutput(self, input):
        input_h, input_w = input.shape[-2:]
        assert input_h % self.kernel_size == 0  
        assert input_w % self.kernel_size == 0
        
        batch_size, channels = input.shape[0], input.shape[1]
        h_out = input_h // self.kernel_size
        w_out = input_w // self.kernel_size
        
        output = input.reshape(batch_size, channels, h_out, self.kernel_size, w_out, self.kernel_size)
        output = output.max(axis=3).max(axis=4)
        
        self.output = output
        return self.output
    
    def updateGradInput(self, input, gradOutput):
        batch_size, channels = input.shape[0], input.shape[1]
        h_out, w_out = gradOutput.shape[2], gradOutput.shape[3]
        
        grad_input = np.zeros_like(input)
        grad_output_reshaped = gradOutput.repeat(self.kernel_size, axis=2).repeat(self.kernel_size, axis=3)
        
        input_reshaped = input.reshape(batch_size, channels, h_out, self.kernel_size, w_out, self.kernel_size)
        max_values = input_reshaped.max(axis=3).max(axis=4)
        max_mask = (input_reshaped == max_values[:,:,:,None,:,None])
        
        grad_input = (grad_output_reshaped * max_mask.reshape(input.shape)).reshape(input.shape)
        self.gradInput = grad_input
        return self.gradInput
    
    def __repr__(self):
        q = 'MaxPool2d, kern %d, stride %d' %(self.kernel_size, self.kernel_size)
        return q

class Flatten(Module):
    def __init__(self):
         super(Flatten, self).__init__()
    
    def updateOutput(self, input):
        self.output = input.reshape(len(input), -1)
        return self.output
    
    def updateGradInput(self, input, gradOutput):
        self.gradInput = gradOutput.reshape(input.shape)
        return self.gradInput
    
    def __repr__(self):
        return "Flatten"