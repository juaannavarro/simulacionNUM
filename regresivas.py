import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

#NOTA# Hemos notado que al aumentar la conductividad se disipa el calor más rápido
##!!!nota!!!: Esto es con Gauss Seiden y lo hacemos para que no converja
#ESTO ES DIFERENCIAS REGRESIVAS



# Entrada de parámetros
N = int(input('Ingresar valor de N: '))
M = int(input('Ingresar valor de M: '))
b = float(input('Ingresar valor de b: ')) # Longitud de la cuerda o barilla
d = float(input('Ingresar valor de d: ')) # Tiempo final

# Cálculo de pasos
h = b / N
k = d / M

#conductivida maxima
conductividad_maxima = h/math.sqrt(2*k)
print(f'La conductividad maxima que soporta es: {conductividad_maxima}')
 #A PARTIR DE ESTA VELOCIDAD SE VUELVE 



# Inicialización de la matriz w con dimensiones correctas
w = np.zeros((N+1, M+1))  # +1 para incluir los bordes


def f(x):
    #return math.exp(-(h*i-b/2)**2)
    '''if 0<x<b/2:
        return 1
    else:
        return 0 #aqui vemos que metemos como dato inicial un pulso es decir que no es diferenciable pero en cambio la solucion sí, luego las soluciones son regularizadas'''
    return math.exp(-(h*i-2.5)**2) 
for i in range(N):
    w[i][0] = 10 * i*(1-i)# Frontera inferior, recordar que xi= xo(a) + ih
    w[i][-1] = -5 #Frontera inferior, recordar que xi= xo(a) + ih
for j in range(M):
    w[0][j] = 5*j #Frontera izquierda, recordar que t_i = jk #Fronetera izquierda (0,t)
    w[N][j] = 5 *(np.sin(2*np.pi*j))  #Frontera derecha (b,t)




for iter in range(100):  # Número de iteraciones
    for j in range(1,M):
        for i in range(1, N):
            #w[i][j] = ((k/h*2)(w[i+1][j] + w[i-1][j]) + (1+h*i)w[i][j-1])/(1+h*i+2(k/h**2))
            #w[i][j]= (-k*(w[i+1][j]+w[i-1][j])-h*2(1+h*i)w[i][j-1])/(-2*k-h2(1+h*i)+k*h**2)
            w[i][j]=((h**3 * i)*(w[i][j+1]+w[i][j-1])) - ((k**3 * j)*(w[i+1][j]+w[i-1][j]))/ 2 *(-j*k**3 + h**3 * i)


# Crear una malla de coordenadas para graficar
x = np.linspace(0, b, N+1)
t = np.linspace(0, d, M+1)
X, T = np.meshgrid(x, t)

# Crear la figura y el eje para la gráfica 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Graficar la superficie
surf = ax.plot_surface(X, T, w.T, cmap='viridis', edgecolor='none')  # Transponer w para que coincida con las dimensiones de X y Y
ax.set_title('Superficie 3D de la Matriz w')
ax.set_xlabel('X')
ax.set_ylabel('T')
ax.set_zlabel('w')

# Añadir barra de colores para la escala
fig.colorbar(surf)
plt.show()