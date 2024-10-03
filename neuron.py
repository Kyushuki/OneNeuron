from math import exp
from random import random, randint
class Neuron:
    def __init__(self):
        self.weight = []
        self.bias = 0
    def activate(self,x):
        return (1/(1+exp(-x)))
    def activate_der(self, x):
        return (x*(1-x))
    def think(self, input):
        net = 0
        for i in range(len(input)):
            net += self.weight[i]*input[i]
        return self.activate(net+self.bias)
class NeuralNetwork:
    def __init__(self):
        self.n1 = Neuron()
        self.temp = 0.01
        self.d = 0
    def train(self, input_set, output_set, epoch):
        # задание смещения и весов 
        self.n1.bias = randint(1,3)
        print(self.n1.bias)
        for i in range(len(input_set[0])):
            self.n1.weight.append(random())
        print(self.n1.weight)
        # начало тренировки по эпохам
        for k in range(epoch):
            # для каждого элемента из набора данных
            for i in range(len(input_set)):
                net = self.n1.think(input_set[i])
                error = output_set[i] - net
                ad = error * self.n1.activate_der(net)
                if k % 1000 == 0:
                    print(ad)
                    print(net)
                for j in range(len(input_set[0])):
                    self.n1.weight[j] += ad*self.temp*input_set[i][j]
                    self.n1.bias += ad*input_set[i][j]
        print(self.n1.weight)
        print("epoch's end")
n = NeuralNetwork()
iset = [[0,1,1],[1,1,1],[0,0,1],[1,0,0]]
oset = [1,0,1,0]
n.train(iset,oset,10000)
print(n.n1.weight)
print(n.n1.bias)
print("должно быть близко к 1")
print(n.n1.think([0,0,1]))
print("должно быть близко к 1")
print(n.n1.think([0,1,1]))
print("должно быть близко к 0")
print(n.n1.think([1,1,1]))
print("должно быть близко к 0")
print(n.n1.think([1,1,0]))
print("должно быть близко к 1")
print(n.n1.think([0,0,0]))