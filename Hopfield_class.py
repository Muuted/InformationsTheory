import numpy as np
import matplotlib.pyplot as plt
from random import randint


class Hopfield_network:
    def __init__(self,I,N):
        self.I = I
        self.sqrt_I = int(np.sqrt(I))
        self.Weights = np.zeros(shape=(I),dtype=int)
        self.a_i = np.zeros(shape=(N,self.I),dtype=int)

        self.D_target = np.zeros(shape=(self.sqrt_I,self.sqrt_I),dtype=int)
        self.J_target = np.zeros(shape=(self.sqrt_I,self.sqrt_I),dtype=int)
        self.C_target = np.zeros(shape=(self.sqrt_I,self.sqrt_I),dtype=int)
        self.M_target = np.zeros(shape=(self.sqrt_I,self.sqrt_I),dtype=int)

        self.D = np.zeros(shape=(self.sqrt_I,self.sqrt_I),dtype=int)
        self.J = np.zeros(shape=(self.sqrt_I,self.sqrt_I),dtype=int)
        self.C = np.zeros(shape=(self.sqrt_I,self.sqrt_I),dtype=int)
        self.M = np.zeros(shape=(self.sqrt_I,self.sqrt_I),dtype=int)

        self.target_matrix_mem_list = [self.D_target,self.J_target,self.C_target,self.M_target]
        self.target_mem_list = []

        self.input_matrix_list = []
        self.input_list = []
        
    def generate_list_of_mem(self):
        self.D_target[0][0], self.D_target[0][1],self.D_target[0][2], self.D_target[0][3] = 1,1,1,1
        self.D_target[1][1], self.D_target[1][4] = 1,1
        self.D_target[2][1], self.D_target[2][4] = 1,1
        self.D_target[3][1], self.D_target[3][4] = 1,1
        self.D_target[4][1],self.D_target[4][2], self.D_target[4][3] = 1,1,1
        
        self.D = self.D_target.copy()
        

        self.J_target[0][:] = 1
        self.J_target[0:4,3] = 1
        self.J_target[3][0] = 1
        self.J_target[4,0:3] = 1
        
        self.J = self.J_target.copy()

        self.C_target[0,1:5] = 1
        self.C_target[1:4,0] = 1
        self.C_target[4,1:5] = 1
        
        self.C = self.C_target.copy()

        self.M_target[0:5,0],self.M_target[0:5,4] = 1,1
        self.M_target[1,1], self.M_target[1,3]= 1,1
        self.M_target[2,2] = 1

        self.M = self.M_target.copy()

        self.input_matrix_list = [self.D,self.J,self.C,self.M]

    def make_list(self,arr):
        placeholderlist = []
        for x in range(self.sqrt_I):
            for y in range(self.sqrt_I):
                placeholderlist.append(arr[x][y])
        
        return placeholderlist
    
    def make_matrix(self):
        pass


    def initial_mem_list(self):
        for n in range(len(self.target_matrix_mem_list)):
            self.target_mem_list.append(self.make_list(self.target_matrix_mem_list[n]))
            self.input_list.append(self.make_list(self.input_matrix_list[n]))


    def initiate_weights(self,start_mem, stop_mem):
        for n in range(start_mem,stop_mem):
            for i in range(self.I):
                for j in range(self.I):
                    self.Weights[i][j] += self.target_mem_list[n][i]*self.target_mem_list[n][j]
                    pass

    def add_noise(self,noise_flip):
        if noise_flip > 0:
            for n in range(self.sqrt_I-1):
                for k in range(noise_flip):
                    i = randint(0,self.sqrt_I-1)
                    j = randint(0,self.sqrt_I-1)
                    
                    if self.input_matrix_list[n][i][j] == 1:
                        self.input_matrix_list[n][i][j] = 0
                    else:
                        self.input_matrix_list[n][i][j] = 1

    def inital_update(self,num_flip):
        self.generate_list_of_mem()

        self.add_noise(noise_flip=num_flip)

        self.initial_mem_list()



    def update_input(self,n):

        """ Calculate all the activities"""
        for i in range(self.sqrt_I):
            for j in range(self.I):
                self.a_i[i] += self.Weights[i][j]*self.input_list[n][j]

        pass






if __name__ == "__main__":
    Hopfield = Hopfield_network(I=25,N=4)

    Hopfield.inital_update(num_flip=0)


    fig,ax = plt.subplots(2,2)
    k = 0
    for i in range(2):
        for j in range(2):
            ax[i,j].matshow(Hopfield.target_matrix_mem_list[k])
            k += 1

    fig,ax = plt.subplots(2,2)
    k = 0
    for i in range(2):
        for j in range(2):
            ax[i,j].matshow(Hopfield.input_matrix_list[k])
            k += 1
    plt.show()