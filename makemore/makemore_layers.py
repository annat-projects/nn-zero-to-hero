import torch

class Linear:
    def __init__(self, fan_in, fan_out, bias=True):
        self.weights = torch.randn((fan_in, fan_out)) / fan_in**0.5
        self.bias = torch.zeros(fan_out) if bias else None
    
    def __call__(self, x):
        self.out = x @ self.weights
        if self.bias is not None:
            self.out += self.bias
        return self.out
    
    def parameters(self):
        return [self.weights] + ([] if self.bias is None else [self.bias])


class BatchNorm1d:
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.training = True
        self.momentum = momentum
        self.eps = eps
        
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        
        self.running_var = torch.ones(dim)
        self.running_mean = torch.zeros(dim)
    
    def __call__(self, x):
        if x.ndim == 2:
            dim = 0
        elif x.ndim == 3:
            dim = (0,1)
            
        if self.training:
            mean = x.mean(dim, keepdims=True)
            var = x.var(dim, keepdims=True)
        else:
            mean = self.running_mean
            var = self.running_var
            
        x_normal = (x - mean) / torch.sqrt(var + self.eps)
        self.out = self.gamma * x_normal + self.beta
        
        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
            
        return self.out
    
    def parameters(self):
        return [self.gamma, self.beta]
    

class Tanh:
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out
    
    def parameters(self):
        return []
    

class Sequential:
    def __init__(self, layers):
        self.layers = layers
        
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return self.out
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
        
class Embedding:
    def __init__(self, emb_num, emb_dim):
        self.weights = torch.randn((emb_num, emb_dim))
        
    def __call__(self, inds):
        self.out = self.weights[inds]
        return self.out
    
    def parameters(self):
        return [self.weights]

class Flatten:
    def __call__(self, x):
        self.out = x.view(x.shape[0], -1)
        return self.out
        
    def parameters(self):
        return []    
    
class FlattenConsecutive:
    def __init__(self, n):
        self.n = n
        
    def __call__(self, x):
        self.out = x.view(x.shape[0], -1, self.n*x.shape[2])
        if self.out.shape[1] == 1:
            self.out = self.out.squeeze(1)
            
        return self.out
        
    def parameters(self):
        return []