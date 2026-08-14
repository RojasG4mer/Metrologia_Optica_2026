import numpy as np
import matplotlib.pyplot as plt
		
		# Evitar envoltura de funciones segun solicitud
		# --- Creacion de la fase a simular ---
x = np.linspace(-10, 10, 512)
y = np.linspace(-10, 10, 512)
X, Y = np.meshgrid(x, y)
		# Simular deformacion (por ejemplo, una Gaussiana que representa elevacion)
fase_real = 15 * np.exp(-(X**2 + Y**2) / 25)
		
		# Variables generales del arreglo
I_dc = 128.0
I_ac = 100.0
		
		# === PARTE 1: Algoritmo de 3 pasos (0, 120, 240 grados) ===
alpha_3 = [0.0, 2*np.pi/3, 4*np.pi/3]
I_3_imgs = [I_dc + I_ac * np.cos(fase_real + a) for a in alpha_3]
		
		# Calculo de fase envuelta 3-steps
		# Ecuacion: arctan( sqrt(3)*(I3-I2) / (2I1-I2-I3) )
num_3 = np.sqrt(3) * (I_3_imgs[2] - I_3_imgs[1])
den_3 = 2 * I_3_imgs[0] - I_3_imgs[1] - I_3_imgs[2]
fase_env_3 = np.arctan2(num_3, den_3)
		
		# === PARTE 2: Algoritmo de 5 pasos (0, 90, 180, 270, 360 grados) ===
alpha_5 = [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]
I_5_imgs = [I_dc + I_ac * np.cos(fase_real + a) for a in alpha_5]
		
		# Calculo de fase envuelta 5-steps
		# Ecuacion: arctan( 2(I4-I2) / (2I3-I1-I5) )
num_5 = 2 * (I_5_imgs[3] - I_5_imgs[1])
den_5 = 2 * I_5_imgs[2] - I_5_imgs[0] - I_5_imgs[4]
fase_env_5 = np.arctan2(num_5, den_5)
		
		# Desenvuelto (Unwrapping con skimage)
from skimage.restoration import unwrap_phase
fase_des_5 = unwrap_phase(fase_env_5)
		
# Visualizacion de las imagenes
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
im0 = axs[0].imshow(I_5_imgs[0], cmap='gray')
axs[0].set_title('Franjas (Paso 0)')
plt.colorbar(im0, ax=axs[0])
		
im1 = axs[1].imshow(fase_env_5, cmap='jet')
axs[1].set_title('Fase Envuelta (5-steps)')
plt.colorbar(im1, ax=axs[1])
		
im2 = axs[2].imshow(fase_des_5, cmap='jet')
axs[2].set_title('Fase Desenvuelta')
plt.colorbar(im2, ax=axs[2])
		
plt.tight_layout()
# plt.show()

# ... [todo el código previo se mantiene igual] ...

plt.tight_layout()

# Reemplazamos plt.show() por plt.savefig() para guardar la imagen
output_path = 'fase_optica_resultado.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close() # Es buena práctica cerrar la figura después de guardarla para liberar memoria