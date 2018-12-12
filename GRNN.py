# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 17:18:25 2018

@author: lj
"""

import numpy as np
import math

def load_data(filename):
    '''导入数据
    input:  file_path(string):训练数据
    output: feature(mat):特征
            label(mat):标签
    '''
    f = open(filename)
    feature = []
    label = []
    for line in f.readlines():
        feature_tmp = []
        lines = line.strip().split('\t')
        for i in range(len(lines)-1):
            feature_tmp.append(float(lines[i]))
        feature.append(feature_tmp)
        label.append(float(lines[-1]))
    return np.mat(feature),np.mat(label).T

def distance(X,Y):
    '''计算两个样本之间的距离
    '''
    return np.sqrt(np.sum(np.square(X-Y),axis = 1))

def distance_mat(trainX,testX):
    '''计算待测试样本与所有训练样本的欧式距离
    input:trainX(mat):训练样本
          testX(mat):测试样本
    output:Euclidean_D(mat):测试样本与训练样本的距离矩阵
    '''
    m,n = np.shape(trainX)
    p = np.shape(testX)[0]
    Euclidean_D = np.mat(np.zeros((p,m)))
    for i in range(p):
        for j in range(m):
            Euclidean_D[i,j] = distance(testX[i,:],trainX[j,:])[0,0]
    return Euclidean_D

def Gauss(Euclidean_D,sigma):
    '''测试样本与训练样本的距离矩阵对应的Gauss矩阵
    input:Euclidean_D(mat):测试样本与训练样本的距离矩阵
          sigma(float):Gauss函数的标准差
    output:Gauss(mat):Gauss矩阵
    '''
    m,n = np.shape(Euclidean_D)
    Gauss = np.mat(np.zeros((m,n)))
    for i in range(m):
        for j in range(n):
            Gauss[i,j] = math.exp(- Euclidean_D[i,j] / (2 * (sigma ** 2)))
    return Gauss

def sum_layer(Gauss,trY):
    '''求和层矩阵，列数等于输出向量维度+1,其中0列为每个测试样本Gauss数值之和
    '''
    m,l = np.shape(Gauss)
    n = np.shape(trY)[1]
    sum_mat = np.mat(np.zeros((m,n+1)))
    ## 对所有模式层神经元输出进行算术求和
    for i in range(m):
        sum_mat[i,0] = np.sum(Gauss[i,:],axis = 1) ##sum_mat的第0列为每个测试样本Gauss数值之和
    ## 对所有模式层神经元进行加权求和
    for i in range(m):             
        for j in range(n):
            total = 0.0
            for s in range(l):
                total += Gauss[i,s] * trY[s,j]
            sum_mat[i,j+1] = total           ##sum_mat的后面的列为每个测试样本Gauss加权之和            
    return sum_mat

def output_layer(sum_mat):
    '''输出层输出
    input:sum_mat(mat):求和层输出矩阵
    output:output_mat(mat):输出层输出矩阵
    '''
    m,n = np.shape(sum_mat)
    output_mat = np.mat(np.zeros((m,n-1)))
    for i in range(n-1):
        output_mat[:,i] = sum_mat[:,i+1] / sum_mat[:,0]
    return output_mat
        
    

if __name__ == '__main__':
    ## 1.导入数据
    print('------------------------1. Load Data----------------------------')
    feature,label = load_data('sine.txt')
    ## 2.数据集和测试集
    print('--------------------2.Train Set and Test Set--------------------')
    trX = feature[0:190,:]
    trY = label[0:190,:]
    teX = feature[190:200,:]
    teY = label[190:200,:]
    ## 3.模式层输出
    print('---------------------3. Output of Hidden Layer------------------')
    Euclidean_D = distance_mat(trX,teX)
    Gauss = Gauss(Euclidean_D,0.1)
    ## 4.求和层输出
    print('---------------------4. Output of Sum Layer---------------------')
    sum_mat = sum_layer(Gauss,trY)
    ## 5.输出层输出
    print('---------------------5. Output of Output Layer------------------')
    output_mat = output_layer(sum_mat)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    