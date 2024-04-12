#!/usr/bin/env python
# coding: utf-8

import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import random
get_ipython().run_line_magic('matplotlib', 'inline')


class Value:
    def __init__(self, value, _children=(), _op='', label=''):
        self.data = value
        self.grad = 0
        self.prev = set(_children)
        self._op = _op
        self.label = label
        self._backward = lambda:None

    def __repr__(self):
        return f"Value (data={self.data})"
    
    def backward(self):
        self.grad = 1.0
        visited = set()
        items = []
        def build_topo(root):
            if root not in visited:
                visited.add(root)
                for child in root.prev:
                    build_topo(child)
                items.append(root)

        build_topo(self)
        items.reverse()
        for item in items:
            item._backward()

    def __radd__(self, other):
        return self + other
    
    def __rsub__(self, other):
        return self - other
    
    def __rmul__(self, other):
        return self * other
    
    def __pow__(self, other):
        assert isinstance(other, (float, int)), "Only supporting int and float"
        res = Value(self.data ** other, (self,), f'**{other}')
        def _backward():
            self.grad += (other * (self.data**(other-1))) * res.grad
        res._backward = _backward
        return res
    
    def __truediv__(self, other):
        print('truediv')
        return self * (other ** -1)
    
    def exp(self):
        res = Value(math.exp(self.data), (self,), 'exp')
        def _backward():
            self.grad += res.data * res.grad
        res._backward = _backward
        return res
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        res = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += res.grad
            other.grad += res.grad
        res._backward = _backward
        return res
    
    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self + (-other)
    
    def __neg__(self):
        return self * -1
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        res = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * res.grad
            other.grad += self.data * res.grad
        res._backward = _backward
        return res
    
    def tanh(self):
        res = Value((math.exp(2*self.data)-1)/(math.exp(2*self.data)+1), (self,), 'tanh')
        def _backward():
            self.grad += (1-(res.data**2)) * res.grad
        res._backward = _backward
        return res
        


#inputs x1, x2
x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')
#weights w1, w2
w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')
#bias b
b = Value(6.88137358, label='b')
# x1w1, x2w2
x1w1 = x1*w1; x1w1.label='x1w1'
x2w2 = x2*w2; x2w2.label='x2w2'
#x1w1 + x2w2 + b
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label='x1w1x2w2'
n = x1w1x2w2 + b; n.label = 'n'

e = (2*n).exp()
eminusone = e-1
eplusone = e+1
o = eminusone / eplusone; o.label='o'
# o = n.tanh()


o.backward()


print(f"x1.grad {x1.grad} x1.data {x1.data}")
print(f"x2.grad {x2.grad} x2.data {x2.data}")
print(f"w1.grad {w1.grad} w1.data {w1.data}")
print(f"w2.grad {w2.grad} w2.data {w2.data}")

print(f"o.grad {o.grad} o.data {o.data}")
print(f"e.grad {e.grad} e.data {e.data}")
print(f"n.grad {n.grad} n.data {n.data}")

print(f"x1w1.grad {x1w1.grad} x1w1.data {x1w1.data}")
print(f"x2w2.grad {x2w2.grad} x2w2.data {x2w2.data}")


# Using PyTorch

x1 = torch.Tensor([2.0]).double(); x1.requires_grad=True
x2 = torch.Tensor([0.0]).double(); x2.requires_grad=True

w1 = torch.Tensor([-3.0]).double(); w1.requires_grad=True
w2 = torch.Tensor([1.0]).double(); w2.requires_grad=True

b = torch.Tensor([6.88137358]).double(); b.requires_grad=True

n = ((x1*w1)+(x2*w2)) + b
o = torch.tanh(n)
o.backward()


x1.grad.item(), x2.grad.item(), w1.grad.item(), w2.grad.item()


class Neuron:
    def __init__(self, nin):
        self.weights = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.bias = Value(random.uniform(-1, 1))
    
    def __call__(self, x):
        act = sum((wi*xi for wi, xi in zip(self.weights, x)), self.bias)
        res = act.tanh()
        return res
    
    def parameters(self):
        res = self.weights + [self.bias]
        return res

class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        res = [n(x) for n in self.neurons]
        return res
    
    def parameters(self):
        res = [p for neuron in self.neurons for p in neuron.parameters()]
        return res
    
class MLP:
    def __init__(self, nin, nouts):
        # nouts = [4, 4, 1]
        # nin = 3
        # produce -> (3,4), (4,4), (4, 1)
        comb_list = [nin] + nouts
        self.layers = [Layer(comb_list[i], comb_list[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x[0] if len(x) == 1 else x
    
    def parameters(self):
        res = [p for layer in self.layers for p in layer.parameters()]
        return res


mlp = MLP(3, [4,4,1])


xs = [
    [2.0, 3.0 , -1.0],
    [3.0, -1.0 , 0.5],
    [0.5, 1.0 , 1.0],
    [1.0, 1.0 , -1.0],
]
ys = [-1.0, 1.0, -0.5, -0.1]


iter = 100
lr = 0.05

for i in range(iter):
    # calc loss
    pred = [mlp(x) for x in xs]
    loss = sum((ygt-ypred)**2  for ygt, ypred in zip(ys, pred))
    if (i%5 ==0):
        print(f"loss -> {loss.data}")
    
    #zero grad
    for p in mlp.parameters():
        p.grad = 0

    # backprop
    loss.backward()

    #update
    for p in mlp.parameters():
        p.data += -lr * p.grad

pred, ys




