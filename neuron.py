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
        self.n2 = Neuron()
        self.n3 = Neuron()
        self.n4 = Neuron()
        self.temp = 0.01
        self.d = 0
    def train(self, input_set, output_set, epoch, neuron : Neuron):
        print("epoch start")
        # задание смещения и весов 
        neuron.bias = randint(1,3)
        for i in range(len(input_set[0])):
            neuron.weight.append(random())
        # начало тренировки по эпохам
        for k in range(epoch):
            # для каждого элемента из набора данных
            for i in range(len(input_set)):
                net = neuron.think(input_set[i])
                error = output_set[i] - net
                ad = error * neuron.activate_der(net)
                # if k % 1000 == 0:
                #     print(ad)
                #     print(net)
                for j in range(len(input_set[0])):
                    neuron.weight[j] += ad*self.temp*input_set[i][j]
                    neuron.bias += ad*input_set[i][j]
        print("epoch's end")
    def train_net(self, input, output):
        self.train(input, output[0], 10000, self.n1)
        self.train(input, output[1], 10000, self.n2)
        self.train(input, output[2], 10000, self.n3)
        self.train(input, output[3], 10000, self.n4)
        print("train over")
    def think(self, input):
        n1 = self.n1.think(input)
        print(n1)
        n2 = self.n2.think(input)
        print(n2)
        n3 = self.n3.think(input)
        print(n3)
        n4 = self.n4.think(input)
        print(n4)
        if n1 >= 0.8:
            print("it's 1")
        if n2 >= 0.8:
            print("it's 2")
        if n3 >= 0.8:
            print("it's 4")
        if n4 >= 0.8:
            print("it's 7")
        if n1 < 0.8 and n2 < 0.8 and n3 < 0.8 and n4 < 0.8:
            print("i dont know such number")

n = NeuralNetwork()
# iset = [[0,1,1],[1,1,1],[0,0,1],[1,0,0]]
# oset = [1,0,1,0]
# n.train(iset,oset,10000)
# print(n.n1.weight)
# print(n.n1.bias)
# print("должно быть близко к 1")
# print(n.n1.think([0,0,1]))
# print("должно быть близко к 1")
# print(n.n1.think([0,1,1]))
# print("должно быть близко к 0")
# print(n.n1.think([1,1,1]))
# print("должно быть близко к 0")
# print(n.n1.think([1,1,0]))
# print("должно быть близко к 1")
# print(n.n1.think([0,0,0]))

# 1 2 4 7
iset = [[0,0,0,1,0,0,0,1,1,0,0,1,0,1,0,1,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0],
[0,1,1,1,0,1,0,0,0,1,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,1,1,1,1],
[1,0,0,1,0,1,0,0,1,0,1,0,0,1,0,1,1,1,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0],
[1,1,1,1,1,0,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0]]
oset = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
n.train_net(iset, oset)
# 1 2 4 7 5 6
iset = [[0,0,0,1,0,0,0,1,1,0,0,1,0,1,0,1,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0],
[0,1,1,1,0,1,0,0,0,1,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,1,1,1,1],
[1,0,0,1,0,1,0,0,1,0,1,0,0,1,0,1,1,1,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0],
[1,1,1,1,1,0,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0],
[1,1,1,1,1,1,0,0,0,0,1,0,0,0,0,1,1,1,1,1,0,0,0,0,1,0,0,0,0,1,1,1,1,1,1],
[1,1,1,1,1,1,0,0,0,0,1,0,0,0,0,1,1,1,1,1,1,0,0,0,1,1,0,0,0,1,1,1,1,1,1]]
for i in range(len(iset)):
    print(i)
    n.think(iset[i])
# oset = [0,0,0,1]
# n.train(iset,oset,10000)
# for i in range(len(iset)):
#     print(n.n1.think(iset[i]))



# [0,0,0,0,0,
#  0,0,0,0,0,
#  0,0,0,0,0,
#  0,0,0,0,0,
#  0,0,0,0,0,
#  0,0,0,0,0,
#  0,0,0,0,0,]