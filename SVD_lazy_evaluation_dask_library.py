#Aplicando librería dask
import matplotlib.pyplot as plt
from scipy import misc, datasets
import pooch
import dask.array as da
import numpy as np

A = datasets.face(gray=True)
fig, ax = plt.subplots(2, 2)
ax = ax.ravel()


def sse_score(X, X_hat):
    return np.sum((X - X_hat)**2)

def svd_dask(X, sv):
    dask_X = da.from_array(X, chunks=(128, 100))
    U, S, Vt = da.linalg.svd_compressed(dask_X, k=sv)
    reconstruction = da.dot(U, da.dot(da.diag(S), Vt))
    computed_reconstruction = reconstruction.compute()
    sse = sse_score(X, computed_reconstruction)
    return computed_reconstruction, sse


singular_values = [50 , 100, 300]

raacom = A

ax[0].imshow(raacom, cmap='gray')
ax[0].set_title(f'Original image, {np.shape(racoon)} pixels',pad = 10)

for plot, elt in zip(range(1,4), singular_values):
    racoon_hat, _= svd_dask(A, sv=elt)
    ax[plot].imshow(racoon_hat, cmap='gray')
    ax[plot].set_title(f'Compression ratio = {round((768*1024)/(768*elt + elt + 1024*elt),2)}',pad = 10)

plt.suptitle('Image Compression with Singular Value Decomposition', fontsize = 16)
plt.tight_layout()
plt.show()

#SSE Plot

def sse_generator():
    A = datasets.face(gray=True)
    for n in range(1, 300):
        _, sse = svd_dask(A, n)
        yield n, sse

x_values, y_values = zip(*sse_generator())
plt.plot(list(x_values), list(y_values))
plt.title("Relation between SV's and SSE")
plt.xlabel("Singular values (SV's)")
plt.ylabel("Error Sum of Squares (SSE)")
plt.show()