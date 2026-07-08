import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# CÓDIGO 1: Patrón de interferencia
# ==========================================
lambda_ = 632e-9
limite = 1e5 * lambda_

# Creación de la malla para el código 1
x_val = np.linspace(-limite, limite, 1024)
y_val = np.linspace(-limite, limite, 1024)
x, y = np.meshgrid(x_val, y_val)

k = 2 * np.pi / lambda_
alpha = 0 * 5e-6          # Inclinación azimuthal del espejo
beta = np.pi / 2 - 5e-6   # Inclinación zenithal del espejo
h = 1 * lambda_

# Ecuación de Intensidad
I = 2 * (1 + np.cos(2 * k * (alpha * x + (np.pi / 2 - beta) * y + h)))

# Graficar primera figura
plt.figure(figsize=(8, 8))
plt.imshow(I, cmap='gray', extent=[-limite, limite, -limite, limite])
plt.axis('equal') 
plt.axis('off')   
plt.title('Patrón de Interferencia')


# ==========================================
# CÓDIGO 2: Desenvolvimiento de fase
# ==========================================
# Crear vectores de x e y de 0 a 256 en pasos de 0.5
x_val_2 = np.arange(0, 256.5, 0.5)
y_val_2 = np.arange(0, 256.5, 0.5)
x2, y2 = np.meshgrid(x_val_2, y_val_2)

# Función de fase original
PHASE = 15 * np.sin(((x2 - 200)**2 + (y2 - 225)**2) / 10000) \
        + 0.002 * ((x2 - 37)**2 + (y2 - 100)**2)

# Función de fase envuelta (Wrapped phase)
FUN = np.exp(1j * PHASE)
WRAP_PHASE = np.angle(FUN)

# Desenvolvimiento de la fase (Unwrapping)
# axis=0 columnas, axis=1 filas
UNWRAP_PHASE = np.unwrap(WRAP_PHASE, axis=0) 
UNWRAP_PHASE = np.unwrap(UNWRAP_PHASE, axis=1)

# Creación de las gráficas 3D
fig = plt.figure(figsize=(18, 5))
ticks = [0, 125, 250]

# Subplot 1: Original Phase
ax1 = fig.add_subplot(131, projection='3d')
surf1 = ax1.plot_surface(x2, y2, PHASE, cmap='viridis', edgecolor='none')
ax1.view_init(elev=52, azim=-30)
ax1.set_title('Original phase (radian)')
ax1.set_xticks(ticks)
ax1.set_yticks(ticks)
fig.colorbar(surf1, ax=ax1, shrink=0.5, pad=0.1)

# Subplot 2: Wrapped Phase
ax2 = fig.add_subplot(132, projection='3d')
surf2 = ax2.plot_surface(x2, y2, WRAP_PHASE, cmap='viridis', edgecolor='none')
ax2.view_init(elev=52, azim=-30)
ax2.set_title('Wrapped phase (radian)')
ax2.set_xticks(ticks)
ax2.set_yticks(ticks)
fig.colorbar(surf2, ax=ax2, shrink=0.5, pad=0.1)

# Subplot 3: Unwrapped Phase
ax3 = fig.add_subplot(133, projection='3d')
surf3 = ax3.plot_surface(x2, y2, UNWRAP_PHASE, cmap='viridis', edgecolor='none')
ax3.view_init(elev=52, azim=-30)
ax3.set_title('Unwrapped phase (radian)')
ax3.set_xticks(ticks)
ax3.set_yticks(ticks)
fig.colorbar(surf3, ax=ax3, shrink=0.5, pad=0.1)

plt.tight_layout()
plt.show()