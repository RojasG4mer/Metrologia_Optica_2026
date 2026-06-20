from PyQt5 import QtWidgets, QtCore, QtGui
import numpy as np
import cv2
import os
# from core.pco_camera import CamaraSimulada # O la cámara que estés usando
from core.camara_web import CamaraWebUSB
from procesamiento.operaciones_basicas import normalizar_float_a_uint8
from procesamiento.filtros_en_frecuencia import filtro_pasa_bajas_frecuencia
from datetime import datetime

class VentanaPrincipal(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Metrología Óptica - Adquisición ESPI")
        self.resize(1000, 700)
        
        # --- Variables de Adquisición y Guardado ---
        self.camara = None
        self.capturando = False
        self.holograma_referencia = None
        self.matriz_cruda_actual = None
        
        # Contadores para el formato de nombres
        self.contador_inicio = 0  # El valor 'x'
        self.contador_toma = 0    # El valor 'y'
        
        # Control de la ráfaga (secuencia)
        self.guardando_secuencia = False
        self.tomas_restantes = 0
        
        # Crear carpeta de guardado automáticamente con Fecha y Hora
        # Ejemplo de formato: capturas_espi_20260618_085320
        marca_tiempo = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.carpeta_destino = f"capturas_espi_{marca_tiempo}"
        os.makedirs(self.carpeta_destino, exist_ok=True)
        
        # Variables de procesamiento
        self.estado_resta = False
        self.estado_filtro = False
        
        # Temporizador para el bucle de video
        self.timer_video = QtCore.QTimer()
        self.timer_video.timeout.connect(self.adquirir_y_procesar)

        self._configurar_ui()
        self._conectar_eventos()

    def _configurar_ui(self):
        widget_central = QtWidgets.QWidget()
        self.setCentralWidget(widget_central)
        layout_principal = QtWidgets.QVBoxLayout(widget_central)
        
        # 1. Layout Superior: Controles de Cámara y Procesamiento
        layout_controles = QtWidgets.QHBoxLayout()
        self.btn_iniciar = QtWidgets.QPushButton("Encender Cámara")
        self.btn_resta = QtWidgets.QPushButton("Ver Resta (ESPI): OFF")
        self.btn_filtro = QtWidgets.QPushButton("Ver Pasa Bajas: OFF")
        
        self.btn_resta.setEnabled(False)
        self.btn_filtro.setEnabled(False)
        
        layout_controles.addWidget(self.btn_iniciar)
        layout_controles.addWidget(self.btn_resta)
        layout_controles.addWidget(self.btn_filtro)
        
        # 2. Layout Inferior: Controles de Guardado (Secuencia)
        layout_guardado = QtWidgets.QHBoxLayout()
        
        self.btn_tomar_inicio = QtWidgets.QPushButton("📸 Tomar Inicio (Referencia)")
        self.btn_tomar_secuencia = QtWidgets.QPushButton("🎞️ Iniciar Secuencia")
        
        # Cuadros de ajuste (SpinBoxes)
        self.spin_cantidad = QtWidgets.QSpinBox()
        self.spin_cantidad.setRange(1, 1000)
        self.spin_cantidad.setValue(100) # Por defecto 100 tomas
        self.spin_cantidad.setPrefix("Tomas: ")
        
        self.spin_velocidad = QtWidgets.QSpinBox()
        self.spin_velocidad.setRange(10, 2000) # De 10ms a 2 segundos
        self.spin_velocidad.setValue(30) # ~33 fps
        self.spin_velocidad.setSuffix(" ms de espera")
        self.spin_velocidad.setToolTip("Velocidad de captura (ms entre cada toma)")
        
        self.btn_tomar_inicio.setEnabled(False)
        self.btn_tomar_secuencia.setEnabled(False)
        self.spin_cantidad.setEnabled(False)
        self.spin_velocidad.setEnabled(False)

        layout_guardado.addWidget(self.btn_tomar_inicio)
        layout_guardado.addWidget(self.spin_cantidad)
        layout_guardado.addWidget(self.spin_velocidad)
        layout_guardado.addWidget(self.btn_tomar_secuencia)

        # 3. Pantalla
        self.label_imagen = QtWidgets.QLabel("Esperando conexión...")
        self.label_imagen.setAlignment(QtCore.Qt.AlignCenter)
        self.label_imagen.setStyleSheet("background-color: black; color: white;")

        # Ensamblar todo
        layout_principal.addLayout(layout_controles)
        layout_principal.addWidget(self.label_imagen, stretch=1)
        layout_principal.addLayout(layout_guardado)

    def _conectar_eventos(self):
        self.btn_iniciar.clicked.connect(self.toggle_camara)
        self.btn_resta.clicked.connect(self.toggle_resta)
        self.btn_filtro.clicked.connect(self.toggle_filtro)
        
        self.btn_tomar_inicio.clicked.connect(self.capturar_inicio)
        self.btn_tomar_secuencia.clicked.connect(self.iniciar_secuencia)
        self.spin_velocidad.valueChanged.connect(self.actualizar_velocidad)

    def actualizar_velocidad(self):
        """Ajusta el reloj interno si el usuario cambia el valor mientras corre."""
        if self.capturando:
            self.timer_video.setInterval(self.spin_velocidad.value())

    def toggle_camara(self):
        if not self.capturando:
            # ------------------------------------------------------------------------------------------------
            # self.camara = CamaraSimulada()
            # self.camara = CamaraSimulada()
            # 1 - para la mac
            # 0 - para la web
            self.camara = CamaraWebUSB(indice_camara=0)
            self.camara.__enter__()
            self.camara.record(number_of_images=1, mode='sequence')
            
            self.capturando = True
            self.btn_iniciar.setText("Apagar Cámara")
            
            # Habilitar botones
            self.btn_resta.setEnabled(True)
            self.btn_filtro.setEnabled(True)
            self.btn_tomar_inicio.setEnabled(True)
            self.spin_cantidad.setEnabled(True)
            self.spin_velocidad.setEnabled(True)
            
            self.timer_video.start(self.spin_velocidad.value())
        else:
            self.timer_video.stop()
            self.capturando = False
            self.camara.__exit__(None, None, None)
            self.btn_iniciar.setText("Encender Cámara")
            
            # Deshabilitar botones
            self.btn_resta.setEnabled(False)
            self.btn_filtro.setEnabled(False)
            self.btn_tomar_inicio.setEnabled(False)
            self.btn_tomar_secuencia.setEnabled(False)
            self.spin_cantidad.setEnabled(False)
            self.spin_velocidad.setEnabled(False)
            self.label_imagen.setText("Cámara apagada.")

    def toggle_resta(self):
        self.estado_resta = not self.estado_resta
        txt = "ON" if self.estado_resta else "OFF"
        self.btn_resta.setText(f"Ver Resta (ESPI): {txt}")

    def toggle_filtro(self):
        self.estado_filtro = not self.estado_filtro
        txt = "ON" if self.estado_filtro else "OFF"
        self.btn_filtro.setText(f"Ver Pasa Bajas: {txt}")

    # --- LÓGICA DE GUARDADO ESPECÍFICA ---
    def capturar_inicio(self):
        """Guarda la imagen base y prepara la secuencia."""
        if self.matriz_cruda_actual is None:
            return
            
        self.contador_inicio += 1
        self.contador_toma = 0 # Reinicia el valor 'y'
        
        # Guarda la matriz en memoria como flotante para las restas matemáticas
        self.holograma_referencia = self.matriz_cruda_actual.astype(np.float32)
        
        # Guarda el archivo TIF crudo
        nombre_archivo = f"inicio_{self.contador_inicio:03d}.tif"
        ruta_completa = os.path.join(self.carpeta_destino, nombre_archivo)
        cv2.imwrite(ruta_completa, self.matriz_cruda_actual)
        
        print(f"Referencia guardada: {nombre_archivo}")
        self.btn_tomar_secuencia.setEnabled(True) # Ahora puedes tomar subsecuentes

    def iniciar_secuencia(self):
        """Activa la bandera para empezar a guardar imágenes en el bucle."""
        if self.holograma_referencia is None:
            return
            
        self.tomas_restantes = self.spin_cantidad.value()
        self.guardando_secuencia = True
        
        # Bloquear botones para no interrumpir
        self.btn_tomar_inicio.setEnabled(False)
        self.btn_tomar_secuencia.setEnabled(False)
        self.spin_cantidad.setEnabled(False)
        print(f"Iniciando captura de {self.tomas_restantes} imágenes...")

    def adquirir_y_procesar(self):
        # 1. Adquisición Cruda
        frame_crudo, _ = self.camara.image()
        self.matriz_cruda_actual = frame_crudo
        
        # 2. Lógica de Guardado en Ráfaga
        if self.guardando_secuencia:
            self.contador_toma += 1
            nombre_toma = f"toma_{self.contador_inicio:03d}_{self.contador_toma:03d}.tif"
            ruta_toma = os.path.join(self.carpeta_destino, nombre_toma)
            
            cv2.imwrite(ruta_toma, self.matriz_cruda_actual)
            self.tomas_restantes -= 1
            
            # Detener ráfaga cuando termine
            if self.tomas_restantes <= 0:
                self.guardando_secuencia = False
                self.btn_tomar_inicio.setEnabled(True)
                self.btn_tomar_secuencia.setEnabled(True)
                self.spin_cantidad.setEnabled(True)
                print(f"Secuencia terminada. Guardado en carpeta: '{self.carpeta_destino}'")

        # 3. Procesamiento Matemático en vivo (No afecta a lo que se guarda)
        imagen_a_mostrar = frame_crudo.astype(np.float32)

        if self.estado_resta and self.holograma_referencia is not None:
            # Ecuación de ESPI: Valor absoluto de (Actual - Referencia)
            resta = imagen_a_mostrar - self.holograma_referencia
            imagen_a_mostrar = np.abs(resta)

        if self.estado_filtro:
            imagen_a_mostrar = filtro_pasa_bajas_frecuencia(imagen_a_mostrar, radio_corte=80)

        # 4. Visualización
        self.mostrar_imagen(imagen_a_mostrar)

    def mostrar_imagen(self, matriz_float):
        img_uint8 = normalizar_float_a_uint8(matriz_float)
        alto, ancho = img_uint8.shape
        q_img = QtGui.QImage(img_uint8.data, ancho, alto, ancho, QtGui.QImage.Format_Grayscale8)
        self.label_imagen.setPixmap(QtGui.QPixmap.fromImage(q_img).scaled(
            self.label_imagen.width(), self.label_imagen.height(), QtCore.Qt.KeepAspectRatio))

    def closeEvent(self, event):
        if self.capturando and self.camara:
            self.timer_video.stop()
            self.camara.__exit__(None, None, None)
        event.accept()