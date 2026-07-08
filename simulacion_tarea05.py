import numpy as np
import matplotlib.pyplot as plt
import sys

print("--- Iniciando el programa de reconstrucción de fase ---")

# 1. Definición de parámetros
print("Generando datos...")
x = np.linspace(-5, 5, 500)
X, Y = np.meshgrid(x, x)
fase_original = 2 * np.pi * np.exp(-(X**2 + Y**2) / 4)

I_dc = 1.0
I_ac = 0.8
alpha1 = np.pi / 4
alpha2 = 3 * np.pi / 4
alpha3 = 5 * np.pi / 4

print("Calculando interferogramas...")
I1 = I_dc + I_ac * np.cos(fase_original + alpha1)
I2 = I_dc + I_ac * np.cos(fase_original + alpha2)
I3 = I_dc + I_ac * np.cos(fase_original + alpha3)

print("Reconstruyendo fase...")
fase_recuperada = np.arctan2(I3 - I2, I1 - I2)

# 2. Verificación en consola
fase_original_wrapped = np.arctan2(np.sin(fase_original), np.cos(fase_original))
error_max = np.max(np.abs(fase_original_wrapped - fase_recuperada))

print(f"--- Resultados ---")
print(f"Fase original en el centro (250,250): {fase_original[250, 250]:.4f}")
print(f"Error absoluto máximo de reconstrucción: {error_max:.2e}")
print("------------------")

# 3. Guardar imagen y mostrar
print("Generando archivo de imagen 'resultados.png'...")
fig, axs = plt.subplots(1, 4, figsize=(16, 4))
axs[0].imshow(I1, cmap='gray'); axs[0].set_title('Interferograma 1')
axs[1].imshow(I2, cmap='gray'); axs[1].set_title('Interferograma 2')
axs[2].imshow(I3, cmap='gray'); axs[2].set_title('Interferograma 3')
axs[3].imshow(fase_recuperada, cmap='jet'); axs[3].set_title('Fase Recuperada')

for ax in axs: ax.axis('off')

# Guardar y forzar el despliegue
plt.savefig('resultados.png', dpi=300)
print("Archivo 'resultados.png' guardado exitosamente.")
plt.show() # Intenta mostrar en pantalla