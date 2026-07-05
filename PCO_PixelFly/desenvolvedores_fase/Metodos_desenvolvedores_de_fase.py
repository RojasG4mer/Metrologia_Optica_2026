import numpy as np

class ProcesadorFase:
    def __init__(self):
        """
        Clase que agrupa métodos de extracción de fase (PSI, Fourier) 
        y métodos de desenvolvimiento (Unwrapping).
        """
        pass

    # ==========================================
    # MÉTODOS DE EXTRACCIÓN DE FASE
    # ==========================================
    def phase_shifting_4_pasos(self, stack_intensidades):
        """
        Recupera la fase envuelta a partir de 4 interferogramas 
        desfasados por pi/2.
        """
        I1, I2, I3, I4 = stack_intensidades[0], stack_intensidades[1], stack_intensidades[2], stack_intensidades[3]
        numerador = I4 - I2
        denominador = I1 - I3
        # arctan2 devuelve la fase en el rango [-pi, pi)
        fase_envuelta = np.arctan2(numerador, denominador)
        return fase_envuelta

    # ==========================================
    # MÉTODOS DE DESENVOLVIMIENTO (UNWRAPPING)
    # ==========================================
    def desenvolvimiento_basico(self, fase_envuelta):
        """
        Desenvolvimiento secuencial rápido (método de Itoh 2D).
        """
        return np.unwrap(np.unwrap(fase_envuelta, axis=0), axis=1)

    def desenvolvimiento_robusto(self, fase_envuelta):
        """
        Desenvolvimiento robusto utilizando scikit-image.
        Ideal para mapas con alto ruido o singularidades topológicas.
        """
        try:
            from skimage.restoration import unwrap_phase
            return unwrap_phase(fase_envuelta)
        except ImportError:
            print("Error: scikit-image no está instalado. Retornando None.")
            return None