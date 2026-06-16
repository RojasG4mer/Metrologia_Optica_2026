import cv2
import numpy as np

def filtro_pasa_bajas_frecuencia(imagen_float, radio_corte=50):
    """
    Aplica un filtro pasa bajas en el dominio de la frecuencia.
    Recibe y devuelve una matriz de tipo float.
    """
    # 1. Transformada de Fourier 2D y centrado de frecuencias
    f_transform = np.fft.fft2(imagen_float)
    f_shift = np.fft.fftshift(f_transform)

    # 2. Crear una máscara circular (1 adentro, 0 afuera)
    filas, columnas = imagen_float.shape
    centro_row, centro_col = filas // 2, columnas // 2
    
    Y, X = np.ogrid[:filas, :columnas]
    distancia_al_centro = np.sqrt((X - centro_col)**2 + (Y - centro_row)**2)
    mascara = distancia_al_centro <= radio_corte

    # 3. Aplicar la máscara y hacer la transformada inversa
    f_shift_filtrado = f_shift * mascara
    f_ishift = np.fft.ifftshift(f_shift_filtrado)
    img_reconstruida = np.fft.ifft2(f_ishift)

    # Devolvemos la magnitud (valores reales)
    return np.abs(img_reconstruida).astype(np.float32)