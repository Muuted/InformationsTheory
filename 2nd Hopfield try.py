import numpy as np
import matplotlib.pyplot as plt
from random import randint

def make_list(arr):
    X,Y = np.shape(arr)
    arr_list = []
    for x in range(X):
        for y in range(Y):
            arr_list.append(arr[x][y])

    return arr_list

def generate_list_of_mem():
    N = 5

    """ --------------Memories------------------ """
    """ D J C M """
    Memories_list = []
    Memories_matrix = []

    D = np.full(shape=(N,N),fill_value=-1,dtype=int)
    D[0][0], D[0][1],D[0][2], D[0][3] = 1,1,1,1
    D[1][1], D[1][4] = 1,1
    D[2][1], D[2][4] = 1,1
    D[3][1], D[3][4] = 1,1
    D[4][1],D[4][2], D[4][3] = 1,1,1
    Memories_list.append(make_list(D))
    Memories_matrix.append(D)
    

    J = np.full(shape=(N,N),fill_value=-1,dtype=int)
    J[0][:] = 1
    J[0:4,3] = 1
    J[3][0] = 1
    J[4,0:3] = 1
    Memories_list.append(make_list(J))
    Memories_matrix.append(J)

    C = np.full(shape=(N,N),fill_value=-1,dtype=int)
    C[0,1:5] = 1
    C[1:4,0] = 1
    C[4,1:5] = 1
    Memories_list.append(make_list(C))
    Memories_matrix.append(C)

    M = np.full(shape=(N,N),fill_value=-1,dtype=int)
    M[0:5,0],M[0:5,4] = 1,1
    M[1,1], M[1,3]= 1,1
    M[2,2] = 1
    Memories_list.append(make_list(M))
    Memories_matrix.append(M)


    return Memories_matrix

def add_noise(arr,num_flip):

    X,Y = np.shape(arr)
    return_arr = arr.copy()

    for k in range(num_flip):
        for x in range(X):
            for y in range(Y):
                i,j = randint(0,X-1), randint(0,Y-1)

                if return_arr[i][j] == 1:
                    return_arr[i][j] = 0
                if return_arr[i][j] == 0:
                    return_arr[i][j] = 1

    return return_arr

class Hopfield:
    def __init__(self,I):
        self.I = I
        self.Weights = np.zeros(shape=(I,I),dtype=int)

        self.matrix_mem_list = []
        self.vec_mem_list = []



    def initiate_hopfield(self,arr):    
    
        self.matrix_mem_list.append(arr)
        list_arr = make_list(arr)
        self.vec_mem_list.append(list_arr)

        for i in range(len(list_arr)):
            for j in range(len(list_arr)):
                self.Weights[i][j] += list_arr[i]*list_arr[j]
                    


def Can_add_new_memories():
    Hopfield_network = Hopfield(I=25)

    memories_matrix_list = generate_list_of_mem()
    N = np.shape(memories_matrix_list)[0]
    for i in range(N-1):
        Hopfield_network.initiate_hopfield(
            arr=memories_matrix_list[i]
        )


    fig, ax = plt.subplots(1,2)
    
    ax[0].matshow(Hopfield_network.Weights)
    ax[0].set_title("Weights materix ,With memories D,J,C")


    Hopfield_network.initiate_hopfield(
            arr=memories_matrix_list[N-1]
        )
    ax[1].matshow(Hopfield_network.Weights)
    ax[1].set_title("Weights materix ,With memories D,J,C and M")
    ax[1].set_ylabel("=>",fontsize=25,rotation="horizontal",ha="right")

    
    plt.show()



def stable_memories():
    Hopfield_network = Hopfield(I=25)

    memories_matrix_list = generate_list_of_mem()
    N = np.shape(memories_matrix_list)[0]
    for i in range(N-1):
        Hopfield_network.initiate_hopfield(
            arr=memories_matrix_list[i]
        )








if __name__ == "__main__":
    Can_add_new_memories()
    #stable_memories()
    