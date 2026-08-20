import numpy as np
# =============================================================================
# MÓDULO 2: PROCESADOR DE FASE (EXTRACCIÓN Y DESENVOLVIMIENTO)
# =============================================================================

class ProcesadorFase:
    def __init__(self):
        """
        Clase que agrupa métodos de extracción de fase (Phase Shifting, Fourier) 
        y métodos de desenvolvimiento (Unwrapping).
        """
        pass

    # ---------------------------------------------------------
    # A. MÉTODOS DE EXTRACCIÓN DE FASE (PHASE SHIFTING)
    # ---------------------------------------------------------
    
    def phase_shifting_3_pasos(self, stack_intensidades):
        """ Algoritmo de medición de fase de 3 medidas. """
        I1, I2, I3 = stack_intensidades[0], stack_intensidades[1], stack_intensidades[2]
        
        numerador = I3 - I2
        denominador = I1 - I2
        return np.arctan2(numerador, denominador)

    def phase_shifting_4_pasos(self, stack_intensidades):
        """ Algoritmo clásico de 4 medidas separadas por pi/2. """
        I1, I2, I3, I4 = stack_intensidades[0], stack_intensidades[1], stack_intensidades[2], stack_intensidades[3]
        
        numerador = I4 - I2
        denominador = I1 - I3
        return np.arctan2(numerador, denominador)

    def phase_shifting_5_pasos_hariharan(self, stack_intensidades):
        """ Algoritmo de Schwider-Hariharan de 5 medidas. """
        I1, I2, I3, I4, I5 = stack_intensidades[0], stack_intensidades[1], stack_intensidades[2], stack_intensidades[3], stack_intensidades[4]
        
        numerador = 2 * (I4 - I2)
        denominador = I1 - 2 * I3 + I5
        return np.arctan2(numerador, denominador)

    def phase_shifting_carre(self, stack_intensidades):
        """ Ecuación de Carré (salto de fase constante pero desconocido). """
        I1, I2, I3, I4 = stack_intensidades[0], stack_intensidades[1], stack_intensidades[2], stack_intensidades[3]
        
        termino_1 = 3 * (I2 - I3) - (I1 - I4)
        termino_2 = (I2 - I3) + (I1 - I4)
        
        # np.abs() previene errores de números imaginarios por ruido numérico
        numerador = np.sqrt(np.abs(termino_1 * termino_2))
        denominador = (I2 + I3) - (I1 + I4)
        
        return np.arctan2(numerador, denominador)

    # ---------------------------------------------------------
    # B. MÉTODO DE EXTRACCIÓN DE FASE (FOURIER - 2 IMÁGENES)
    # ---------------------------------------------------------
    
    def obtener_espectro_fourier(self, interferograma):
        """ Paso 1: Obtener el espectro para visualización interactiva. """
        F = np.fft.fftshift(np.fft.fft2(interferograma))
        espectro_log = np.log(1 + np.abs(F))
        return F, espectro_log

    def _obtener_c_xy(self, interferograma, x1, y1, x2, y2):
        """ 
        Función interna: Extrae el número complejo c(x,y) de una imagen 
        aplicando el filtro RECTANGULAR en las coordenadas indicadas.
        """
        F = np.fft.fftshift(np.fft.fft2(interferograma))
        filas, columnas = F.shape
        centro_f, centro_c = filas // 2, columnas // 2

        # 1. Crear máscara rectangular basada en los clics
        Y, X = np.ogrid[:filas, :columnas]
        mascara = (X >= x1) & (X <= x2) & (Y >= y1) & (Y <= y2)

        F_filtrada = np.zeros_like(F)
        F_filtrada[mascara] = F[mascara]

        # 2. Desplazar el lóbulo al centro
        pico_f = (y1 + y2) // 2
        pico_c = (x1 + x2) // 2
        desplazamiento_f = centro_f - pico_f
        desplazamiento_c = centro_c - pico_c
        
        F_centrada = np.roll(F_filtrada, shift=(desplazamiento_f, desplazamiento_c), axis=(0, 1))

        # 3. Transformada Inversa
        c_xy = np.fft.ifft2(np.fft.ifftshift(F_centrada))
        return c_xy

    def fase_fourier_dos_imagenes(self, I_referencia, I_deformada, x1, y1, x2, y2):
        """
        Paso 2: Aplica la ecuación de diferencia de fase usando 
        el estado base (referencia) y el estado deformado.
        """
        # Obtenemos c1(x,y) y c2(x,y)
        c1 = self._obtener_c_xy(I_referencia, x1, y1, x2, y2)
        c2 = self._obtener_c_xy(I_deformada, x1, y1, x2, y2)

        # Separamos partes reales e imaginarias
        R1, I1 = np.real(c1), np.imag(c1)
        R2, I2 = np.real(c2), np.imag(c2)

        # Ecuación para Delta Phi
        numerador = (R1 * I2) - (I1 * R2)
        denominador = (R1 * R2) + (I1 * I2)

        delta_phi_envuelta = np.arctan2(numerador, denominador)
        return delta_phi_envuelta

    # ---------------------------------------------------------
    # C. MÉTODOS DE DESENVOLVIMIENTO (UNWRAPPING)
    # ---------------------------------------------------------
    
    def desenvolvimiento_basico(self, fase_envuelta):
        """ Desarrollado secuencial rápido (método de Itoh 2D). """
        return np.unwrap(np.unwrap(fase_envuelta, axis=0), axis=1)

    def desenvolvimiento_robusto(self, fase_envuelta):
        try:
            from skimage.restoration import unwrap_phase
            return unwrap_phase(fase_envuelta)
        except ImportError:
            # Desenvolvimiento 2D simplificado de respaldo
            return np.unwrap(np.unwrap(fase_envuelta, axis=0), axis=1)