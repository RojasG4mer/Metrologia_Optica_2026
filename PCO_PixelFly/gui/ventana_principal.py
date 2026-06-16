from PyQt5 import QtWidgets, QtCore, QtGui
import numpy as np
from core.pco_camera import CamaraSimulada
from procesamiento.operaciones_basicas import restar_consecutivas, normalizar_float_a_uint8
from procesamiento.filtros_en_frecuencia import filtro_pasa_bajas_frecuencia

class VentanaPrincipal(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sistema de Adquisición y Procesamiento")
        self.resize(900, 600)
        
        # Variables de estado y almacenamiento
        self.camara = None
        self.frame_anterior = None
        
        self.capturando = False
        self.estado_resta = False
        self.estado_filtro = False
        
        # Temporizador para el tiempo real (ej. 30 ms = ~33 fps)
        self.timer_video = QtCore.QTimer()
        self.timer_video.timeout.connect(self.adquirir_y_procesar)

        self._configurar_ui()
        self._conectar_eventos()

    def _configurar_ui(self):
        widget_central = QtWidgets.QWidget()
        self.setCentralWidget(widget_central)
        layout_principal = QtWidgets.QVBoxLayout(widget_central)
        layout_botones = QtWidgets.QHBoxLayout()

        # Botones
        self.btn_iniciar = QtWidgets.QPushButton("Iniciar Captura")
        self.btn_resta = QtWidgets.QPushButton("Activar Resta: OFF")
        self.btn_filtro = QtWidgets.QPushButton("Activar Pasa Bajas: OFF")
        
        # Deshabilitar botones de procesamiento hasta que empiece la captura
        self.btn_resta.setEnabled(False)
        self.btn_filtro.setEnabled(False)

        layout_botones.addWidget(self.btn_iniciar)
        layout_botones.addWidget(self.btn_resta)
        layout_botones.addWidget(self.btn_filtro)
        
        self.label_imagen = QtWidgets.QLabel("Esperando hardware...")
        self.label_imagen.setAlignment(QtCore.Qt.AlignCenter)

        layout_principal.addLayout(layout_botones)
        layout_principal.addWidget(self.label_imagen, stretch=1)

    def _conectar_eventos(self):
        self.btn_iniciar.clicked.connect(self.toggle_captura)
        self.btn_resta.clicked.connect(self.toggle_resta)
        self.btn_filtro.clicked.connect(self.toggle_filtro)

    def toggle_captura(self):
        if not self.capturando:
            # Encender cámara y temporizador
            self.camara = CamaraSimulada()
            # Para cuando ya tenga la camara conectada ---------------------------------------------------------------------------------
            # self.camara = pco.Camera()
            self.camara.__enter__() # Abrimos conexión
            self.camara.record(number_of_images=1, mode='sequence') # Preparamos buffer
            self.capturando = True
            self.btn_iniciar.setText("Detener Captura")
            self.btn_resta.setEnabled(True)
            self.btn_filtro.setEnabled(True)
            self.timer_video.start(30) # Ejecutar cada 30 ms
        else:
            # Apagar cámara y temporizador
            self.timer_video.stop()
            self.capturando = False
            if self.camara:
                self.camara.__exit__(None, None, None)
            self.btn_iniciar.setText("Iniciar Captura")
            self.btn_resta.setEnabled(False)
            self.btn_filtro.setEnabled(False)
            self.label_imagen.setText("Captura detenida")

    def toggle_resta(self):
        self.estado_resta = not self.estado_resta
        estado_txt = "ON" if self.estado_resta else "OFF"
        self.btn_resta.setText(f"Activar Resta: {estado_txt}")
        # Reiniciar el frame anterior si apagamos la resta para no usar imágenes viejas
        if not self.estado_resta:
            self.frame_anterior = None

    def toggle_filtro(self):
        self.estado_filtro = not self.estado_filtro
        estado_txt = "ON" if self.estado_filtro else "OFF"
        self.btn_filtro.setText(f"Activar Pasa Bajas: {estado_txt}")

    def adquirir_y_procesar(self):
        """Este método se ejecuta continuamente como un bucle de video."""
        # 1. Adquisición pura (convertida a float inmediatamente)
        frame_crudo, _ = self.camara.image()
        frame_actual = frame_crudo.astype(np.float32)
        
        imagen_a_mostrar = frame_actual

        # 2. Lógica de Resta
        if self.estado_resta:
            if self.frame_anterior is not None:
                imagen_a_mostrar = restar_consecutivas(frame_actual, self.frame_anterior)
            # Guardamos el frame actual puro para la siguiente iteración
            self.frame_anterior = frame_actual 

        # 3. Lógica de Filtrado Frecuencial
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
        """Asegura que la cámara se apague si el usuario cierra la ventana de golpe."""
        if self.capturando and self.camara:
            self.timer_video.stop()
            self.camara.__exit__(None, None, None)
        event.accept()