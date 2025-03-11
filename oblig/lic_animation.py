import h5py
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os


def visualize(arr: np.array, cmap: str = "viridis"):
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
    X = np.clip(X, 0, a.shape[0] - 2)
    Y = np.clip(Y, 0, a.shape[1] - 2)
    
    U1 = (1.0 - fracX) * a[X + 0, Y + 0] + fracX * a[X + 1, Y + 0]
    U2 = (1.0 - fracX) * a[X + 0, Y + 1] + fracX * a[X + 1, Y + 1]
    U = (1.0 - fracY) * U1 + fracY * U2
    return U


# Load data
d1 = h5py.File("../isabel_2d.h5", "r")
u = np.array(d1["Velocity"]["X-comp"])
v = np.array(d1["Velocity"]["Y-comp"])
mag = magnitude2(u, v)
mesh_size = u.shape[0]
visualize(magnitude2(u, v))

# Generate noise texture
n = np.random.random(u.shape)

# Vectorized coordinates/points
i = np.arange(0, mesh_size).astype(float)
j = np.arange(0, mesh_size).astype(float)
ii, jj = np.meshgrid(i, j)

# Step sizes
l_values = np.linspace(100, 6_000, num=20, dtype=int)
output_dir = "/Users/edvart/Programming/visualization/animations/lic/ar"
os.makedirs(output_dir, exist_ok=True)

step_size = 0.005

for frame_idx, l in enumerate(l_values):
    print(f"Processing frame {frame_idx + 1}/{len(l_values)} with l={l}")
    
    # kernel 
    h = np.exp(-np.linspace(-2, 2, l)**2)  
    h /= h.sum()
    
    # result texture (accumulated noise)
    F = np.zeros_like(n)
    F = h[0] * interp2d(n, ii, jj)
    
    # fwd
    xx, yy = ii, jj
    for k in range(1, l):
        u_local = interp2d(u, xx, yy)
        v_local = interp2d(v, xx, yy)
        mag_local = np.sqrt(u_local**2 + v_local**2) + 1e-32  

        dtx = step_size * u_local / mag_local
        dty = step_size * v_local / mag_local

        xx += dtx
        yy += dty

        F += h[k] * interp2d(n, xx, yy)
    
    # bwd
    xx, yy = ii, jj
    for k in range(1, l):
        u_local = interp2d(u, xx, yy)
        v_local = interp2d(v, xx, yy)
        mag_local = np.sqrt(u_local**2 + v_local**2) + 1e-6  

        dtx = step_size * u_local / mag_local
        dty = step_size * v_local / mag_local

        xx -= dtx
        yy -= dty

        F += h[k] * interp2d(n, xx, yy)

    # normalize LIC result
    F_proc = np.copy(F.T)
    F_proc[mag < 0.01] = 0.0
    F_proc -= F_proc.min()
    F_proc /= F_proc.max()

    # histogram equalization
    F_eq = np.copy(F_proc)
    F_eq *= 255
    F_eq = F_eq.astype(np.uint8)
    F_eq = cv2.equalizeHist(F_eq)
    F_eq = (F_eq / 255.0).astype(np.float32)

    mag_proc = np.copy(mag)
    mag_proc -= mag_proc.min()
    mag_proc /= mag_proc.max()

    cmap = plt.cm.RdYlGn(mag_proc)
    cmap[..., :3] *= F_eq[..., np.newaxis]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_facecolor("black")
    ax.set_axis_off()
    plt.imshow(cmap, cmap='grey')
    plt.savefig(os.path.join(output_dir, f"frame_{frame_idx:03d}.png"), bbox_inches='tight', pad_inches=0)
    plt.close(fig)

print("Simulation complete. Frames saved in 'simulation_frames' directory.")