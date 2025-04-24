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

def make_matrix(arr):
    N,I = np.shape(arr)
    rows = int(np.sqrt(I))
    matrix_list = []
    for n in range(N):
        matrix = np.zeros(shape=(rows,rows),dtype=int)
        i = 0
        for x in range(rows):
            for y in range(rows):
                matrix[x][y] = arr[n][i]
                i += 1
        matrix_list.append(matrix.copy())
    return matrix_list

def generate_list_of_mem():
    N = 5

    """ --------------Memories------------------ """
    """ D J C M """
    Memories_list = []
    Memories_matrix = []

    D = np.zeros(shape=(N,N),dtype=int)
    D[0][0], D[0][1],D[0][2], D[0][3] = 1,1,1,1
    D[1][1], D[1][4] = 1,1
    D[2][1], D[2][4] = 1,1
    D[3][1], D[3][4] = 1,1
    D[4][1],D[4][2], D[4][3] = 1,1,1
    Memories_list.append(make_list(D))
    Memories_matrix.append(D)
    

    J = np.zeros(shape=(N,N),dtype=int)
    J[0][:] = 1
    J[0:4,3] = 1
    J[3][0] = 1
    J[4,0:3] = 1
    Memories_list.append(make_list(J))
    Memories_matrix.append(J)

    C = np.zeros(shape=(N,N),dtype=int)
    C[0,1:5] = 1
    C[1:4,0] = 1
    C[4,1:5] = 1
    Memories_list.append(make_list(C))
    Memories_matrix.append(C)

    M = np.zeros(shape=(N,N),dtype=int)
    M[0:5,0],M[0:5,4] = 1,1
    M[1,1], M[1,3]= 1,1
    M[2,2] = 1
    Memories_list.append(make_list(M))
    Memories_matrix.append(M)


    return Memories_list,Memories_matrix

def Hebb_rule_Hopfield(memories,Hopfield):
    N,I = np.shape(memories)
    
    for n in range(N):
        for i in range(I):
            for j in range(I):
                Hopfield[i][j] += memories[n][i]*memories[n][j]

    
def add_noise(arr_matrix,num_flip):
    N,X,Y= np.shape(arr_matrix)

    for k in range(num_flip):
        for n in range(N):
            i = randint(0,X-1)
            j = randint(0,Y-1)
            
            if arr_matrix[n][i][j] == 1:
                arr_matrix[n][i][j] = 0
            else:
                arr_matrix[n][i][j] = 1
            
    
    
def Correcting_error(Hopfield,mem):

    pass



if __name__ == "__main__":
    Memories_list,Memories_matrix = generate_list_of_mem()
    I = len(Memories_list[0])
    Hopfield = np.zeros(shape=(I,I),dtype=int)
    
    fig,ax = plt.subplots(2,2)

    add_noise(Memories_matrix,num_flip=3)
    Memories_list = []
    k = 0
    for i in range(2):
        for j in range(2):
            ax[i,j].matshow(Memories_matrix[k])
            k += 1
    

    fig,ax = plt.subplots(2,2)
    matrix = make_matrix(arr=Memories_list)


    k = 0
    for i in range(2):
        for j in range(2):
            ax[i,j].matshow(matrix[k])
            k += 1
    

    Hebb_rule_Hopfield(
        memories=Memories_list[0:3]
        ,Hopfield=Hopfield
    )


    plt.matshow(Hopfield)
    plt.colorbar()

    
    plt.show()