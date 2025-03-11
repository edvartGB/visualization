import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def visualize(arr: np.array):
    img = np.copy(arr)
    img = img - img.min()
    img = img / img.max()
    plt.imshow(img)


def magnitude2(arrX: np.array, arrY: np.array):
    return np.sqrt(arrX**2 + arrY**2)


def interp2(u, v, x, y):
    X = np.floor(x).astype(int)
    Y = np.floor(y).astype(int)
    fracX = x - X
    fracY = y - Y

    A = np.array([u[X + 0, Y + 0], v[X + 0, Y + 0]])
    B = np.array([u[X + 1, Y + 0], v[X + 1, Y + 0]])
    C = np.array([u[X + 0, Y + 1], v[X + 0, Y + 1]])
    D = np.array([u[X + 1, Y + 1], v[X + 1, Y + 1]])
    U1 = (1.0 - fracX) * A + fracX * B
    U2 = (1.0 - fracX) * C + fracX * D
    U = (1.0 - fracY) * U1 + fracY * U2
    return U


def interp2d(a, x, y):
    X = np.floor(x).astype(int)
    Y = np.floor(y).astype(int)
    fracX = x - X
    fracY = y - Y

    U1 = (1.0 - fracX) * a[X + 0, Y + 0] + fracX * a[X + 1, Y + 0]
    U2 = (1.0 - fracX) * a[X + 0, Y + 1] + fracX * a[X + 1, Y + 1]
    U = (1.0 - fracY) * U1 + fracY * U2
    return U




d1 = h5py.File("../isabel_2d.h5", "r")
u = np.array(d1["Velocity"]["X-comp"])
v = np.array(d1["Velocity"]["Y-comp"])
img = np.array([u, v, np.zeros_like(u)])


num_particles = 1000 
T = 30
dt = 0.1
steps = int(T / dt)

trajectories = []
lines = []
seeds_x = []
seeds_y = []

mesh_size = u.shape[0]

fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(0, u.shape[0])
ax.set_ylim(0, u.shape[0])
ax.set_facecolor("black")

# generate seeds
particles_per_dim = np.ceil(np.sqrt(num_particles)).astype(int)
print(particles_per_dim)
meshgrid = np.meshgrid(np.linspace(0, mesh_size, particles_per_dim), np.linspace(0, mesh_size, particles_per_dim))
seeds_x = meshgrid[0].flatten()
seeds_y = meshgrid[1].flatten()

for i in range(num_particles):
    pos_x, pos_y = seeds_x[i], seeds_y[i]
    trajectory_x = [pos_x]
    trajectory_y = [pos_y]

    for j in range(steps):
        out_of_bounds = (
            (pos_x < 0)
            or (pos_y < 0)
            or (pos_x >= u.shape[0] - 2)
            or (pos_y >= u.shape[1] - 2)
        )
        if out_of_bounds:
            break
        else:
            tmp_x = pos_x
            pos_x += interp2d(u, pos_x, pos_y) * dt
            pos_y += interp2d(v, tmp_x, pos_y) * dt
            trajectory_x.append(pos_x)
            trajectory_y.append(pos_y)

    trajectories.append((trajectory_x, trajectory_y))
    (line,) = ax.plot([], [], "w", linewidth=0.5)
    lines.append(line)


def update(frame):
    for i, line in enumerate(lines):
        x_data = trajectories[i][0][:frame]
        y_data = trajectories[i][1][:frame]
        line.set_data(x_data, y_data)
    return lines


plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.axis('off')
visualize(magnitude2(u.T, v.T))
ani = animation.FuncAnimation(fig, update, frames=steps, interval=10, blit=True)
ani.save("../animations/uniform_seeding_isabel.mp4")
plt.show()
