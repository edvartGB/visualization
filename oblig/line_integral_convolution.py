import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.ndimage import gaussian_filter


def visualize(arr: np.array, cmap: str = "viridis", flip: bool = False):
    img = np.copy(arr)
    img = img - img.min()
    img = img / img.max()
    plt.imshow(img, cmap=cmap, origin="upper")


def magnitude2(arrX: np.array, arrY: np.array):
    return np.sqrt(arrX**2 + arrY**2)


def interp2d(a, x, y):
    X = np.floor(x).astype(int)
    Y = np.floor(y).astype(int)
    fracX = x - X
    fracY = y - Y
    X = min(max(X, 0), a.shape[0] - 2)
    Y = min(max(Y, 0), a.shape[1] - 2)

    U1 = (1.0 - fracX) * a[X + 0, Y + 0] + fracX * a[X + 1, Y + 0]
    U2 = (1.0 - fracX) * a[X + 0, Y + 1] + fracX * a[X + 1, Y + 1]
    U = (1.0 - fracY) * U1 + fracY * U2
    return U




d1 = h5py.File("../isabel_2d.h5", "r")
u = np.array(d1["Velocity"]["X-comp"])
v = np.array(d1["Velocity"]["Y-comp"])
img = np.array([u, v, np.zeros_like(u)])

mesh_size = u.shape[0]

# generate noise texture
n = np.random.random(u.shape)
lic_result = np.zeros(u.shape)

# perform LIC
l = 10
dt = 0.5
h = [1.0]*l
kernel_magnitude = sum(h)


for i in range(mesh_size):
    for j in range(mesh_size):
        F_ij = 0
        # fwd
        pos_x = i
        pos_y = j

        for k in range(l):
            tmp_x = pos_x
            pos_x += interp2d(u, pos_x, pos_y) * dt
            pos_y += interp2d(v, tmp_x, pos_y) * dt
            F_ij += h[k]*interp2d(n, pos_x, pos_y)

        # bwd 
        pos_x = i
        pos_y = j
        for k in range(l):
            tmp_x = pos_x
            pos_x -= interp2d(u, pos_x, pos_y) * dt
            pos_y -= interp2d(v, tmp_x, pos_y) * dt
            F_ij += h[k]*interp2d(n, pos_x, pos_y)
        F_ij /= (2*kernel_magnitude)
        lic_result[i, j] = F_ij
    if i % 10 == 0:
        print(f"{i}/{mesh_size}")




fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(0, u.shape[0])
ax.set_ylim(0, u.shape[0])
ax.set_facecolor("black")

plt.axis('off')

result = lic_result*magnitude2(u, v)
visualize(lic_result.T, cmap='grey')



plt.show()
