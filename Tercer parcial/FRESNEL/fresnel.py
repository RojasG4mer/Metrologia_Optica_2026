import numpy as np
import cv2
import matplotlib.pyplot as plt

def reconstruir_holograma_fresnel(ruta_imagen, d, lam, dx, dy):
    """
    Reconstruye un holograma digital utilizando la aproximación discreta de difracción de Fresnel.
    
    Parámetros:
    - ruta_imagen: str, ruta al archivo del holograma.
    - d: float, distancia de propagación en metros.
    - lam: float, longitud de onda del láser en metros.
    - dx, dy: float, tamaño físico del píxel del CCD en metros.
    """
    
    # 1. Cargar el holograma [ A(x_h, y_h) ]
    # Se carga en escala de grises y se convierte a flotante para evitar desbordamientos
    holograma = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
    if holograma is None:
        raise ValueError(f"No se pudo cargar la imagen en la ruta: {ruta_imagen}")
    
    holograma = holograma.astype(np.float64)
    M, N = holograma.shape  # M filas (y), N columnas (x)
    
    # 2. Crear las coordenadas espaciales discretas en el plano del holograma
    # Se centran en cero para mantener la simetría al aplicar la FFT
    x = (np.arange(N) - N / 2) * dx
    y = (np.arange(M) - M / 2) * dy
    X, Y = np.meshgrid(x, y)
    
    # 3. Haz de referencia conjugado [ R*(x_h, y_h) ]
    # Asumimos una onda plana incidiendo normalmente (R=1).
    # Si tu montaje es 'off-axis' (fuera de eje), deberás introducir aquí 
    # la portadora espacial con el ángulo respectivo.
    R_conj = np.ones((M, N), dtype=np.complex128)
    
    # 4. Calcular la Función Chirp (Factor exponencial cuadrático)
    # exp[ i * pi / (lambda * d) * (x_h^2 + y_h^2) ]
    fase_esferica = np.exp(1j * (np.pi / (lam * d)) * (X**2 + Y**2))
    
    # 5. Modulación del campo en el plano del holograma
    # Multiplicamos la transmitancia del holograma por la referencia y el chirp
    campo_modulado = holograma * R_conj * fase_esferica
    
    # 6. Propagación numérica mediante Transformada Rápida de Fourier (FFT 2D)
    # fftshift centra las frecuencias espaciales nulas en el origen
    U = np.fft.fftshift(np.fft.fft2(campo_modulado))
    
    # 7. Extracción de los resultados físicos
    intensidad = np.abs(U)**2
    # Aplicamos un logaritmo para comprimir el rango dinámico y mejorar 
    # la visualización de los órdenes de difracción.
    intensidad_log = np.log(1 + intensidad)
    
    # Fase óptica envuelta entre [-pi, pi]
    fase = np.angle(U)
    
    return holograma, intensidad_log, fase

# ==========================================
# Ejecución del algoritmo
# ==========================================
if __name__ == "__main__":
    # Parámetros físicos (Ajustables según tu montaje optomecatrónico de laboratorio)
    lam_laser = 532e-9     # Longitud de onda Nd:YAG (532 nm) en metros
    distancia = 0.425      # Distancia de enfoque d = 42.5 cm en metros
    tam_pixel = 5.2e-6     # Tamaño de pixel del CCD (5.2 um) en metros
    
    # Ruta de tu holograma (asegúrate de que apunte a tu imagen capturada)
    ruta_holograma = 'images/Holograma.png' 
    
    try:
        # Ejecutar propagación
        holograma_base, espectro_intensidad, mapa_fase = reconstruir_holograma_fresnel(
            ruta_imagen=ruta_holograma, 
            d=distancia, 
            lam=lam_laser, 
            dx=tam_pixel, 
            dy=tam_pixel
        )
        
        # Configurar la figura de Matplotlib para visualizar resultados
        plt.figure(figsize=(18, 6))
        
        # Gráfica 1: Holograma de entrada
        plt.subplot(1, 3, 1)
        plt.title('Holograma Digital Original')
        plt.imshow(holograma_base, cmap='gray')
        plt.axis('off')
        
        # Gráfica 2: Reconstrucción de Intensidad
        plt.subplot(1, 3, 2)
        plt.title(f'Intensidad Reconstruida (d = {distancia*100} cm)')
        plt.imshow(espectro_intensidad, cmap='gray')
        plt.axis('off')
        
        # Gráfica 3: Reconstrucción de Fase
        plt.subplot(1, 3, 3)
        plt.title('Fase Óptica Envuelta $\phi$')
        # El colormap 'hsv' o 'jet' son idóneos para visualizar ciclos de fase de -pi a pi
        im_fase = plt.imshow(mapa_fase, cmap='hsv') 
        plt.colorbar(im_fase, fraction=0.046, pad=0.04)
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error durante el procesamiento: {e}")