import numpy as np
import cv2

def restar_consecutivas(frame_actual, frame_anterior):
    """Resta dos matrices forzando el formato flotante."""
    img_act = frame_actual.astype(np.float32)
    img_ant = frame_anterior.astype(np.float32)
    
    return img_act - img_ant

def normalizar_float_a_uint8(imagen_float):
    """Convierte la matriz flotante resultante a 8 bits (0-255) para visualización."""
    return cv2.normalize(imagen_float, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)