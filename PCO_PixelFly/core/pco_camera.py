import pco
import cv2
import numpy as np
import time

class CamaraSimulada:
    def __init__(self):
        # Simulamos el nombre del modelo
        self.camera_name = "PCO.PixelFly (SIMULADA)"

    def __enter__(self):
        print("[Simulador] Conectando a la cámara virtual...")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("[Simulador] Desconectando cámara virtual de forma segura.")

    def record(self, number_of_images=1, mode='sequence'):
        print(f"[Simulador] Preparando captura de {number_of_images} imagen(es)...")
        # Un pequeño retraso para simular el tiempo del hardware
        time.sleep(0.1) 

    def image(self):
        # La PCO PixelFly típicamente tiene una resolución de 1392 x 1040
        ancho, alto = 1392, 1040
        
        # Generamos una matriz de ruido aleatorio. 
        # Como las cámaras científicas suelen ser de 12 o 14 bits, 
        # generamos valores hasta 4095 en tipo uint16.
        ruido = np.random.randint(0, 4095, (alto, ancho), dtype=np.uint16)
        
        # Agregamos un pequeño gradiente suave de fondo para que parezca luz
        x = np.linspace(0, 1000, ancho)
        y = np.linspace(0, 1000, alto)
        xv, yv = np.meshgrid(x, y)
        gradiente = (xv * 0.3 + yv * 0.3).astype(np.uint16)
        
        imagen_falsa = np.clip(ruido + gradiente, 0, 4095)
        
        # Simulamos los metadatos que devuelve la librería original
        metadatos = {'width': ancho, 'height': alto, 'camera': 'Simulador'}
        
        return imagen_falsa, metadatos

# ¡Este es tu interruptor maestro!
# Cambia a False cuando vayas al laboratorio.
MODO_SIMULACION = True 

def probar_conexion():
    print("Iniciando sistema de visión...")
    
    try:
        # Decidimos qué "cámara" instanciar
        if MODO_SIMULACION:
            camara_activa = CamaraSimulada()
        else:
            camara_activa = pco.Camera()

        # El bloque 'with' funciona idéntico para la real y la simulada
        with camara_activa as cam:
            print(f"Conectado: {cam.camera_name}")
            
            cam.record(number_of_images=1, mode='sequence')
            imagen, metadatos = cam.image()
            
            print(f"Captura exitosa. Resolución: {imagen.shape}")
            
            # Normalizamos para visualizar con OpenCV
            img_normalizada = cv2.normalize(imagen, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            
            cv2.imshow("Prueba de Visión", img_normalizada)
            print("Presiona cualquier tecla en la imagen para salir...")
            
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
    except Exception as e:
        print(f"\nError en el sistema: {e}")

if __name__ == "__main__":
    probar_conexion()