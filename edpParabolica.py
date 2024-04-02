import numpy as np
import math
import matplotlib.pyplot as plt

# Definición de variables
b = 5
d = 10
N = 40
M = 400
v = 0.3  # Selecciona un valor dentro del rango 0.2 < v < 1.5
h = b / N
k = d / M

v_max = h / (2*k)**0.5


def f(x):
    return math.exp(-(x - b / 2)**2)

# Inicialización de la matriz w
w = np.zeros((M + 1, N + 1))

# Condiciones de contorno
for j in range(1, M):
    w[j][0] = 0
    w[j][N] = 0

# Condiciones iniciales
for i in range(1, N):
    w[0][i] = f(i * h)

# Método de diferencias finitas
for j in range(M):
    for i in range(1, N ):  # Ajustar el rango de i
        w[j + 1][i] = (1 - 2 * k * v**2 / h**2) * w[j][i] + (k * v**2 / h**2) * (w[j][i + 1] + w[j][i - 1])

# Visualización en 3D
x = np.linspace(0, b, N + 1)
t = np.linspace(0, d, M + 1)
X, T = np.meshgrid(x, t)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Graficar la superficie
surf = ax.plot_surface(X, T, w, cmap='viridis', edgecolor='none')  
ax.set_title('Superficie 3D de la Matriz w')
ax.set_xlabel('X')
ax.set_ylabel('T')
ax.set_zlabel('w')
plt.show()




        


    