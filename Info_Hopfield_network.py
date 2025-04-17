import numpy as np
import matplotlib.pyplot as plt


HopField = np.zeros(shape=(25,25))


""" --------------Memories------------------ """
""" D J C M """
N = 5
D = np.zeros(shape=(N,N))
D[0][0], D[0][1],D[0][2], D[0][3] = 1,1,1,1
D[1][1], D[1][4] = 1,1
D[2][1], D[2][4] = 1,1
D[3][1], D[3][4] = 1,1
D[4][1],D[4][2], D[4][3] = 1,1,1

#plt.matshow(D)

J = np.zeros(shape=(N,N))
J[0][:] = 1
J[0:4,3] = 1
J[3][0] = 1
J[4,0:3] = 1

C = np.zeros(shape=(N,N))
C[0,1:5] = 1
C[1:4,0] = 1
C[4,1:5] = 1

M = np.zeros(shape=(N,N))
M[0:5,0],M[0:5,4] = 1,1
M[1,1], M[1,3]= 1,1
M[2,2] = 1

fig,ax = plt.subplots()
ax[0,0].matshow(D)
ax[0,1].matshow(J)
ax[1,0].matshow(C)
ax[1,1].matshow(M)
plt.show()