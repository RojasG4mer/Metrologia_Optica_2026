import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.restoration import unwrap_phase
import sys

def cargar_imagenes_fpp(ruta_base, tipo_carga="objeto"):
    """
    Ingresa a las subcarpetas '1', '2', '3' y '4' dentro de la ruta especificada
    y carga el primer archivo .tif válido para FPP que encuentre en cada una.
    """
    imagenes = []
    for i in range(1, 5):
        subcarpeta = Path(ruta_base) / str(i)
        
        if not subcarpeta.exists():
            print(f"Error: La carpeta {subcarpeta} no existe.")
            sys.exit(1)
            
        archivos_tif = list(subcarpeta.glob("*.tif"))
        if not archivos_tif:
            print(f"Error: No se encontró ningún archivo .tif en la carpeta: {subcarpeta}")
            sys.exit(1)
            
        archivos_validos = [f for f in archivos_tif if f.suffix == '.tif']
        if not archivos_validos:
             print(f"Error: No se encontró archivo .tif válido en: {subcarpeta}")
             sys.exit(1)
             
        archivos_validos.sort()
        archivo_seleccionado = archivos_validos[0]
        
        print(f"Cargando {tipo_carga.upper()} (paso {i}): {archivo_seleccionado.name}")
        
        img = cv2.imread(str(archivo_seleccionado), cv2.IMREAD_UNCHANGED)
        
        if img is None:
            print(f"Error al leer la imagen: {archivo_seleccionado}")
            sys.exit(1)
            
        imagenes.append(img.astype(np.float32))
        
    return imagenes

# ==========================================
# 1. Configuración de Rutas
# ==========================================
directorio_script = Path(__file__).parent
directorio_base = directorio_script / "proyeccion de franjas 01"
ruta_objeto = directorio_base / "OBJETO"
ruta_referencia = directorio_base / "REFERENCIA"

if not directorio_base.exists():
    print(f"ERROR CRÍTICO: No se encuentra la carpeta base '{directorio_base.absolute()}'.")
    sys.exit(1)

# ==========================================
# 2. PARÁMETROS DE CALIBRACIÓN FÍSICA (mm)
# ==========================================
# DATOS REALES DE LA PRÁCTICA (Convertidos de cm a mm para estándar de metrología)
L_dist = 171.0       # Separación proyector-cámara (base 'd' = 17.1 cm)
Lc_dist = 670.0      # Cámara-plano de referencia (67.0 cm)
p_x = 27.0           # Periodo de las franjas sobre el plano ('p' = 2.7 cm)

# Nota sobre la escala lateral (X y Y):
# Para calcular esto con precisión geométrica en mm, necesitas saber el campo de 
# visión (Field of View) de tu cámara. Ejemplo: Si tu cámara ve un área de 300 mm 
# de ancho y la imagen tiene 1000 píxeles, la escala es 300/1000 = 0.3 mm/píxel.
# He colocado 0.1 mm como estimación genérica por ahora.
escala_xy = 0.1      

# ==========================================
# 3. Carga de Imágenes
# ==========================================
I_obj = cargar_imagenes_fpp(ruta_objeto, "objeto")
I_ref = cargar_imagenes_fpp(ruta_referencia, "referencia")

# ==========================================
# 4. Cálculo de la Fase y Modulación
# ==========================================
num_obj = I_obj[3] - I_obj[1]
den_obj = I_obj[0] - I_obj[2]
fase_obj = np.arctan2(num_obj, den_obj)
modulacion_obj = np.sqrt(num_obj**2 + den_obj**2)

num_ref = I_ref[3] - I_ref[1]
den_ref = I_ref[0] - I_ref[2]
fase_ref = np.arctan2(num_ref, den_ref)

# ==========================================
# 5. Máscara Dinámica
# ==========================================
umbral_dinamico = 0.15 * np.max(modulacion_obj)
mascara_cruda = (modulacion_obj > umbral_dinamico).astype(np.uint8)
kernel = np.ones((7,7), np.uint8)
mascara_limpia = cv2.morphologyEx(mascara_cruda, cv2.MORPH_OPEN, kernel).astype(bool)

# ==========================================
# 6. Diferencia de Fase
# ==========================================
diferencia_fase = np.arctan2(np.sin(fase_obj - fase_ref), np.cos(fase_obj - fase_ref))
diferencia_fase_enmascarada = np.ma.array(diferencia_fase, mask=~mascara_limpia)

# ==========================================
# 7. Desenvolvimiento (Unwrapping)
# ==========================================
print("Ejecutando unwrapping seguro...")
fase_desenvuelta = unwrap_phase(diferencia_fase_enmascarada)

fase_final = fase_desenvuelta.data.copy()
fase_final[~mascara_limpia] = np.nan

# ==========================================
# 8. CONVERSIÓN A UNIDADES FÍSICAS (mm)
# ==========================================
print("Convirtiendo radianes a milímetros usando datos experimentales...")
# h(x,y) = (delta_phi * p_x) / (2 * pi * tan(alpha))
# tan(alpha) = L / LC
tan_alpha = L_dist / Lc_dist
altura_z_mm = (fase_final * p_x) / (2 * np.pi * tan_alpha)

# Ajuste de signo (si el relieve queda invertido como hoyo en vez de cúpula, cambiar a 1)
altura_z_mm = altura_z_mm * -1 

# ==========================================
# 9. Visualización de Resultados Físicos
# ==========================================
fig2d, axs = plt.subplots(1, 3, figsize=(18, 5))

axs[0].imshow(fase_obj, cmap='hsv')
axs[0].set_title('Fase Envuelta')
axs[0].axis('off')

axs[1].imshow(mascara_limpia, cmap='gray')
axs[1].set_title('Máscara de Modulación')
axs[1].axis('off')

im = axs[2].imshow(altura_z_mm, cmap='viridis')
axs[2].set_title('Topografía Desenvuelta (2D)')
axs[2].axis('off')
fig2d.colorbar(im, ax=axs[2], fraction=0.046, pad=0.04, label='Altura Z [mm]')
fig2d.canvas.manager.set_window_title('Análisis 2D - Físico')
fig2d.tight_layout()

# --- VENTANA 3D CON EJES EN MILÍMETROS ---
fig3d = plt.figure(figsize=(9, 7))
ax3 = fig3d.add_subplot(111, projection='3d')
paso = 4 
Y_pix, X_pix = np.indices(altura_z_mm.shape)

# Convertir píxeles X, Y a milímetros usando la escala
X_mm = X_pix * escala_xy
Y_mm = Y_pix * escala_xy

ax3.plot_surface(X_mm[::paso, ::paso], Y_mm[::paso, ::paso], altura_z_mm[::paso, ::paso], 
                 cmap='viridis', edgecolor='none', rstride=1, cstride=1)
ax3.set_title('Topografía 3D (Escala Real)')

ax3.set_xlabel('Eje X [mm]', labelpad=10)
ax3.set_ylabel('Eje Y [mm]', labelpad=10)
ax3.set_zlabel('Altura Z [mm]', labelpad=10)

ax3.view_init(elev=40, azim=-45)
fig3d.canvas.manager.set_window_title('Visor 3D Físico')
fig3d.tight_layout()

plt.show()