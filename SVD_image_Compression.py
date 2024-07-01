# TODO Dibuja las graficas

import matplotlib.pyplot as plt
from scipy import misc, datasets
import pooch
import numpy as np



#Funciones
def sse_score(X, X_hat):
    return np.sum((X - X_hat)**2) 

def svm(X):
    U,S,Vt = np.linalg.svd(X, full_matrices = False)
    S = np.diag(S)
    return U, S, Vt # S es una matriz diagonal

def reconstruction(U, S, Vt):
    return np.dot(U,np.dot(S, Vt))

def image_compression(A, n_comp):
    # TODO 1: Aplicar SVD (usando la función que hemos creado)
    U, S, Vt = svm(A)
    
    # TODO 2: Reconstruir usando solo el número de componentes n_comp (usando la función que hemos creado)
    U_reducida = U[:,0:n_comp]
    S_reducida = S[0:n_comp,0:n_comp]
    Vt_reducida = Vt[0:n_comp,:]
    A_hat = reconstruction(U_reducida, S_reducida, Vt_reducida)
    
    # TODO 3: Calcular el error
    sse = sse_score(A, A_hat)

    return A_hat, sse # A_hat es la matriz comprimida y sse es su error respecto de A



# 1 Crear gráfica con plt.figure()
fig, ax = plt.subplots(2, 2)
A = datasets.face(gray=True)

# 2 Elegir un n_comp y aplicar la función image_compression() y
# 3 Usar plt.imshow(A_hat, cmap=plt.cm.gray), donde A_hat va a ser la matriz comprimida resultante del paso anterior
n_comp = 10
racoon_hat, sse = image_compression(A, n_comp)
ax[0,1].imshow(racoon_hat, cmap='gray')
ax[0,1].set_title(f'Compression ratio = {round((768*1024)/(768*n_comp + n_comp + 1024*n_comp),2)}',pad = 10)

n_comp = 50
racoon_hat, sse = image_compression(A, n_comp)
ax[1,0].imshow(racoon_hat, cmap='gray')
ax[1,0].set_title(f'Compression ratio = {round((768*1024)/(768*n_comp + n_comp + 1024*n_comp),2)}',pad = 10)

n_comp = 300
racoon_hat, sse = image_compression(A, n_comp)
ax[1,1].imshow(racoon_hat, cmap='gray')
ax[1,1].set_title(f'Compression ratio = {round((768*1024)/(768*n_comp + n_comp + 1024*n_comp),2)}',pad = 10)


racoon = A
ax[0,0].imshow(racoon, cmap='gray')
ax[0,0].set_title(f'Original image, {np.shape(racoon)} pixels',pad = 10)
#racoon = misc.face(gray=True)

# 4 Añadir un título a la gráfica
plt.suptitle('Image Compression with Singular Value Decomposition', fontsize = 16)
plt.tight_layout()
plt.show()
# Repetir para distintas compresiones (distinto n_comp)