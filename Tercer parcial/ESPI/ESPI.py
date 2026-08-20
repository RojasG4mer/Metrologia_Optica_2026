import numpy as np
import cv2
import matplotlib.pyplot as plt

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

def simular_o_cargar_imagenes():
    """
    Carga imágenes reales utilizando rutas absolutas dinámicas para evitar 
    errores de lectura en OpenCV.
    """
    # 1. Obtiene la ruta de la carpeta exacta donde está guardado este archivo (ESPI.py)
    directorio_actual = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Construye las rutas absolutas usando los nombres REALES de tu captura
    ruta_ref = os.path.join(directorio_actual, 'holograma_0081.tif')      # <--- CORREGIDO
    ruta_modo = os.path.join(directorio_actual, 'holograma_0082.tif')
    
    # 3. Carga las imágenes
    img_ref = cv2.imread(ruta_ref, cv2.IMREAD_GRAYSCALE)
    img_modo = cv2.imread(ruta_modo, cv2.IMREAD_GRAYSCALE)
    
    # 4. Validaciones de seguridad (Si falla, te dirá exactamente qué ruta intentó leer)
    if img_ref is None:
        raise FileNotFoundError(f"¡Error! OpenCV no encontró la imagen de referencia. Buscó exactamente en:\n{ruta_ref}\nVerifica que el archivo exista ahí y que sea .tif y no .tiff")
        
    if img_modo is None:
        raise FileNotFoundError(f"¡Error! OpenCV no encontró la imagen del modo. Buscó exactamente en:\n{ruta_modo}\nVerifica el nombre del archivo.")
        
    return img_ref, img_modo


def procesar_y_visualizar_espi():
    # 1. Obtener las imágenes (Referencia y Modo 1)
    img_ref, img_modo = simular_o_cargar_imagenes()

    # 2. Procesamiento ESPI (Resta absoluta y filtrado de speckle)
    ref_f = img_ref.astype(np.float32)
    modo_f = img_modo.astype(np.float32)
    
    franjas_crudas = np.abs(modo_f - ref_f)
    franjas_norm = cv2.normalize(franjas_crudas, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Filtrado para eliminar el ruido láser puro y rescatar las franjas de correlación
    franjas_filtradas = cv2.medianBlur(franjas_norm, 7)
    franjas_filtradas = cv2.GaussianBlur(franjas_filtradas, (7, 7), 0)

    # 3. Extracción de fase por Transformada de Fourier (FTP)
    F = np.fft.fftshift(np.fft.fft2(franjas_filtradas))
    filas, cols = franjas_filtradas.shape
    centro_x, centro_y = cols // 2, filas // 2
    
    mascara = np.zeros((filas, cols), dtype=np.uint8)
    mascara[0:centro_y, :] = 1 # Filtrar semiplano superior
    cv2.circle(mascara, (centro_x, centro_y), 15, 0, -1) # Anular componente DC
    
    F_filtrada = F * mascara
    I_filtrada = np.fft.ifft2(np.fft.ifftshift(F_filtrada))
    fase_envuelta = np.angle(I_filtrada)

    # 4. Visualización con Matplotlib (Lado izquierdo: Speckle/Franjas, Lado derecho: Modo / Fase)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Lado Izquierdo: Patrón de Franjas de Correlación ESPI (Speckle procesado)
    im0 = axes[0].imshow(franjas_filtradas, cmap='gray')
    axes[0].set_title("Franjas de Correlación ESPI (Speckle)", fontsize=13, fontweight='bold')
    axes[0].axis('off')
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Lado Derecho: Fase Envuelta del Modo de Vibración
    im1 = axes[1].imshow(fase_envuelta, cmap='hsv')
    axes[1].set_title("Fase Envuelta Extraída (Modo 1)", fontsize=13, fontweight='bold')
    axes[1].axis('off')
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    plt.suptitle("Análisis Óptico ESPI - Interferometría de Moteado", fontsize=15, y=0.98)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Ejecutando procesamiento ESPI y generando visualización...")
    procesar_y_visualizar_espi()