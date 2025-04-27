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
    I = np.shape(arr)
    rows = int(np.sqrt(I))
    matrix = np.zeros(shape=(rows,rows),dtype=int)
    i = 0
    for x in range(rows):
        for y in range(rows):
            matrix[x][y] = arr[i]
            i += 1
    
    return matrix

def generate_list_of_mem():
    N = 5
    init_val = -1
    """ --------------Memories------------------ """
    """ D J C M """
    Memories_list = []
    Memories_matrix = []

    D = np.full(shape=(N,N),fill_value=init_val,dtype=int)
    D[0][0], D[0][1],D[0][2], D[0][3] = 1,1,1,1
    D[1][1], D[1][4] = 1,1
    D[2][1], D[2][4] = 1,1
    D[3][1], D[3][4] = 1,1
    D[4][1],D[4][2], D[4][3] = 1,1,1
    Memories_list.append(make_list(D))
    Memories_matrix.append(D)
    

    J = np.full(shape=(N,N),fill_value=init_val,dtype=int)
    J[0][:] = 1
    J[0:4,3] = 1
    J[3][0] = 1
    J[4,0:3] = 1
    Memories_list.append(make_list(J))
    Memories_matrix.append(J)

    C = np.full(shape=(N,N),fill_value=init_val,dtype=int)
    C[0,1:5] = 1
    C[1:4,0] = 1
    C[4,1:5] = 1
    Memories_list.append(make_list(C))
    Memories_matrix.append(C)

    M = np.full(shape=(N,N),fill_value=init_val,dtype=int)
    M[0:5,0],M[0:5,4] = 1,1
    M[1,1], M[1,3]= 1,1
    M[2,2] = 1
    Memories_list.append(make_list(M))
    Memories_matrix.append(M)


    return Memories_matrix, Memories_list

def add_noise(arr,num_flip):

    X,Y = np.shape(arr)
    return_arr = arr.copy()

    for k in range(num_flip):
        i,j = randint(0,X-1), randint(0,Y-1)
        return_arr[i][j] *= -1

    return return_arr

class Hopfield:
    def __init__(self,I):
        self.I = I
        self.Weights = np.zeros(shape=(I,I),dtype=int)

        self.matrix_mem_list = []
        self.vec_mem_list = []

    def Brain_damge(self,percent):
        damaged_neurons = int(percent*(self.I)**2)

        for _ in range(damaged_neurons):
            i = randint(0,self.I-1)
            j = randint(0,self.I-1)
            self.Weights[i][j] = 0


    def initiate_hopfield(self,arr):    
    
        self.matrix_mem_list.append(arr)
        list_arr = make_list(arr)
        self.vec_mem_list.append(list_arr)

        for i in range(len(list_arr)):
            for j in range(len(list_arr)):
                self.Weights[i][j] += list_arr[i]*list_arr[j]
        
        for i in range(len(list_arr)):
            self.Weights[i][i] = 0
                    
    def activity(self,arr_list):
        a = np.zeros(self.I)
        for i in range(self.I):
            for j in range(self.I):
                a[i] += self.Weights[i][j]*arr_list[j]
        
        return a
    
    def activity_rule(self,arr_list):
        a = self.activity(arr_list=arr_list)
        return_arr = np.zeros(self.I)
        for i in range(self.I):
            if a[i] >= 0:
                return_arr[i] = 1
            if a[i] < 0:
                return_arr[i] = -1

        return return_arr

def correction_of_noise(arr,Hopfield_network):
    input = [arr.copy()]
    k = 0
    Change = True
    while Change == True:
        updated_arr = Hopfield_network.activity_rule(
            arr_list= input[k]
        )
        k += 1

        Change = False
        for i in range(len(input[0])):
            if input[k-1][i] != updated_arr[i]:
                Change = True                
        
        if Change == True:
            input.append(updated_arr.copy())
        if Change == False:
            break
        if k > 30:
            print(f"too many iterations , k={k}")
            exit()

    return input

def Can_add_new_memories():
    Hopfield_network = Hopfield(I=25)

    memories_matrix_list, memories_list = generate_list_of_mem()
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



def stable_memories(flipped_bits):
    Hopfield_network = Hopfield(I=25)

    memories_matrix_list, memories_list = generate_list_of_mem()

    N = np.shape(memories_matrix_list)[0]
    N += 1
    
    for i in range(N-1):
        Hopfield_network.initiate_hopfield(
            arr=memories_matrix_list[i]
        )

    noisy_letters = []
    matrix_correction = []
    target_letters_list = ["D","J","C","M"]
    k = 0    
    for n in range(N-1):
        noisy_letters = []
        matrix_correction = []
        noisy_matrix = add_noise(
            arr=memories_matrix_list[n]
            ,num_flip=flipped_bits
            )
        #print(np.shape(memories_matrix_list[n]))

        noisy_arr = make_list(arr=noisy_matrix)
        #print(np.shape(noisy_arr))
        updated_arr =Hopfield_network.activity_rule(
            arr_list=noisy_arr
        )       
        #print(np.shape(updated_arr))

        change_input = correction_of_noise(
            arr=noisy_arr
            ,Hopfield_network=Hopfield_network
            )

        noisy_letters.append(change_input.copy())

        changed_matrix = [ ]
        for k in range(len(change_input)):
            changed_matrix.append(
                make_matrix(arr=change_input[k])
            )
        
        NN = len(changed_matrix)

        fig, ax = plt.subplots(1,NN)
        #fig.get_current_fig_manager().window.showMaximized()
        fig.canvas.manager.window.showMaximized()
        fig.suptitle(
            f"Target letter ={target_letters_list[n]} with Noise level ={flipped_bits} bits flipped"
            , fontsize=25
            )
        for i in range(NN):
            ax[i].matshow(changed_matrix[i])
            if i == 0:
                ax[i].set_ylabel(
                    f"Noisy {target_letters_list[n]}          "
                    ,rotation="horizontal",fontsize=25
                    )
                ax[i].set_title(f"Initial noisy letter, noise level={flipped_bits}")
            if  i == NN-1:
                ax[i].set_title("final result after correction")
        
        plt.draw()
        plt.pause(0.5 )
        fig.savefig("C:\\Users\\AdamSkovbjergKnudsen\\Desktop\\Informations teori\\"
                     + f"noise={flipped_bits}bits and Target={target_letters_list[n]}"
                     )
        
        #plt.show()
    plt.close()




def Brain_damage_memories(flipped_bits,Brain_damage_percent):
    Hopfield_network = Hopfield(I=25)

    memories_matrix_list, memories_list = generate_list_of_mem()

    N = np.shape(memories_matrix_list)[0]
    N += 1
    
    for i in range(N-1):
        Hopfield_network.initiate_hopfield(
            arr=memories_matrix_list[i]
        )

    Hopfield_network.Brain_damge(percent=Brain_damage_percent)
    noisy_letters = []
    matrix_correction = []
    target_letters_list = ["D","J","C","M"]
    k = 0    
    for n in range(N-1):
        noisy_letters = []
        matrix_correction = []
        noisy_matrix = add_noise(
            arr=memories_matrix_list[n]
            ,num_flip=flipped_bits
            )
        #print(np.shape(memories_matrix_list[n]))

        noisy_arr = make_list(arr=noisy_matrix)
        #print(np.shape(noisy_arr))
        updated_arr =Hopfield_network.activity_rule(
            arr_list=noisy_arr
        )       
        #print(np.shape(updated_arr))

        change_input = correction_of_noise(
            arr=noisy_arr
            ,Hopfield_network=Hopfield_network
            )

        noisy_letters.append(change_input.copy())

        changed_matrix = [ ]
        for k in range(len(change_input)):
            changed_matrix.append(
                make_matrix(arr=change_input[k])
            )
        
        NN = len(changed_matrix)

        fig, ax = plt.subplots(1,NN)
        #fig.get_current_fig_manager().window.showMaximized()
        fig.canvas.manager.window.showMaximized()
        fig.suptitle(
            f"Target letter ={target_letters_list[n]} with Noise level ={flipped_bits} bits flipped"
            , fontsize=25
            )
        for i in range(NN):
            ax[i].matshow(changed_matrix[i])
            if i == 0:
                ax[i].set_ylabel(
                    f"Noisy {target_letters_list[n]}          "
                    ,rotation="horizontal",fontsize=25
                    )
                ax[i].set_title(f"Initial noisy letter, noise level={flipped_bits}")
            if  i == NN-1:
                ax[i].set_title("final result after correction")
        
        plt.draw()
        plt.pause(0.5 )
        fig.savefig("C:\\Users\\AdamSkovbjergKnudsen\\Desktop\\Informations teori\\"
                    + f"brain damage ={int(Brain_damage_percent*100)} % "
                    + f"noise={flipped_bits}bits and Target={target_letters_list[n]}"
                     )
        
        
       
        plt.close()



if __name__ == "__main__":
    #Can_add_new_memories()
    flip_list = [1,3,5,10,20]
    for i in flip_list:
        #stable_memories(flipped_bits=i)
        print(i)
        Brain_damage_memories(
            flipped_bits=i
            ,Brain_damage_percent=0.1
        )
    