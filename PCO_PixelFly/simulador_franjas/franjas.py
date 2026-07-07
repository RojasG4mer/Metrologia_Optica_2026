import numpy as np

class SimuladorFranjas:
    def __init__(self, resolucion=512, I_obj=0.5, I_ref=0.5):
        """
        Inicializa el simulador definiendo la resolución espacial y 
        las intensidades de los haces de interferencia.
        """
        self.N = resolucion
        self.I_obj = I_obj
        self.I_ref = I_ref
        
        self.x = np.linspace(-1, 1, self.N)
        self.y = np.linspace(-1, 1, self.N)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        # Coordenadas polares
        self.R = np.sqrt(self.X**2 + self.Y**2)
        self.Theta = np.arctan2(self.Y, self.X)
        
    def _calcular_salida(self, fase_absoluta, alpha=0.0):
        """
        Calcula el interferograma utilizando el modelo riguroso de dos ondas.
        alpha: Cambio de fase extra (para Phase Shifting).
        """
        # Ecuación de interferencia
        termino_cruzado = 2 * np.sqrt(self.I_obj * self.I_ref) * np.cos(fase_absoluta + alpha)
        intensidad = self.I_obj + self.I_ref + termino_cruzado
        
        # La fase envuelta teórica (Ground Truth) de la diferencia de fase original
        fase_envuelta = np.angle(np.exp(1j * fase_absoluta))
        
        return intensidad, fase_envuelta

    def generar_lineales(self, fx=15, alpha=0.0):
        fase = 2 * np.pi * fx * self.X
        return self._calcular_salida(fase, alpha)

    def generar_circulares(self, k=60, alpha=0.0):
        fase = k * (self.R**2)
        return self._calcular_salida(fase, alpha)

    def generar_espiral(self, k=40, m=4, alpha=0.0):
        fase = k * self.R + m * self.Theta
        return self._calcular_salida(fase, alpha)
        
    def generar_stack_phase_shifting(self, tipo_fase='espiral', pasos=4, **kwargs):
        """
        Método de conveniencia para generar un cubo de datos (stack) con N pasos de fase.
        Ideal para alimentar directamente algoritmos de PSI de 3, 4 o 5 pasos.
        """
        alphas = np.linspace(0, 2*np.pi, pasos, endpoint=False)
        stack_intensidades = []
        
        for a in alphas:
            if tipo_fase == 'lineal':
                I, _ = self.generar_lineales(alpha=a, **kwargs)
            elif tipo_fase == 'circular':
                I, _ = self.generar_circulares(alpha=a, **kwargs)
            elif tipo_fase == 'espiral':
                I, _ = self.generar_espiral(alpha=a, **kwargs)
                
            stack_intensidades.append(I)
            
        # Retornamos el stack [N, Alto, Ancho] y el arreglo de alphas utilizados
        return np.array(stack_intensidades), alphas