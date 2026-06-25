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
        
        # 1. Layout Principal Horizontal (Izquierda: Imagen, Derecha: Controles)
        layout_principal = QtWidgets.QHBoxLayout(widget_central)

        # --- PANEL IZQUIERDO (VISUALIZACIÓN) ---
        self.label_imagen = QtWidgets.QLabel("Esperando conexión...")
        self.label_imagen.setAlignment(QtCore.Qt.AlignCenter)
        self.label_imagen.setStyleSheet("background-color: black; color: white; border: 1px solid gray;")
        self.label_imagen.setMinimumSize(640, 480) # Evita que la ventana colapse al iniciar

        # --- PANEL DERECHO (CONTROLES) ---
        # Creamos un layout vertical para apilar los botones
        layout_derecho = QtWidgets.QVBoxLayout()
        layout_derecho.setContentsMargins(15, 10, 15, 10) # Da un poco de margen a los bordes
        layout_derecho.setSpacing(10) # Espacio entre cada botón

        # Sección: Hardware y Procesamiento en Vivo
        label_hardware = QtWidgets.QLabel("<b>Control de Cámara y Filtros</b>")
        label_hardware.setAlignment(QtCore.Qt.AlignCenter)
        
        self.btn_iniciar = QtWidgets.QPushButton("Encender Cámara")
        # ... (código existente) ...
        self.btn_resta = QtWidgets.QPushButton("Ver Resta (ESPI): OFF")
        self.btn_filtro = QtWidgets.QPushButton("Ver Pasa Bajas: OFF")
        
        # NUEVO: Control dinámico para el tamaño del filtro
        self.label_radio = QtWidgets.QLabel("Radio de Corte: 80 px")
        self.label_radio.setAlignment(QtCore.Qt.AlignCenter)
        
        self.slider_filtro = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_filtro.setRange(1, 500)   # Rango de apertura de la máscara circular
        self.slider_filtro.setValue(80)       # Valor inicial
        
        # Deshabilitar botones y slider al inicio
        self.btn_resta.setEnabled(False)
        self.btn_filtro.setEnabled(False)
        self.slider_filtro.setEnabled(False)  # NUEVO

        layout_derecho.addWidget(label_hardware)
        layout_derecho.addWidget(self.btn_iniciar)
        layout_derecho.addWidget(self.btn_resta)
        layout_derecho.addWidget(self.btn_filtro)
        layout_derecho.addWidget(self.label_radio)   # NUEVO
        layout_derecho.addWidget(self.slider_filtro) # NUEVO
    
        # Separador visual entre secciones
        linea = QtWidgets.QFrame()
        linea.setFrameShape(QtWidgets.QFrame.HLine)
        linea.setFrameShadow(QtWidgets.QFrame.Sunken)
        layout_derecho.addWidget(linea)

        # Sección: Captura ESPI
        label_espi = QtWidgets.QLabel("<b>Secuencia de Adquisición</b>")
        label_espi.setAlignment(QtCore.Qt.AlignCenter)
        
        self.btn_tomar_inicio = QtWidgets.QPushButton("📸 Tomar Inicio (Referencia)")
        
        self.spin_cantidad = QtWidgets.QSpinBox()
        self.spin_cantidad.setRange(1, 1000)
        self.spin_cantidad.setValue(100)
        self.spin_cantidad.setPrefix("Tomas: ")
        
        self.spin_velocidad = QtWidgets.QSpinBox()
        self.spin_velocidad.setRange(10, 2000)
        self.spin_velocidad.setValue(30)
        self.spin_velocidad.setSuffix(" ms de espera")
        self.spin_velocidad.setToolTip("Velocidad de captura (ms entre cada toma)")
        
        self.btn_tomar_secuencia = QtWidgets.QPushButton("🎞️ Iniciar Secuencia")
        
        self.btn_tomar_inicio.setEnabled(False)
        self.btn_tomar_secuencia.setEnabled(False)
        self.spin_cantidad.setEnabled(False)
        self.spin_velocidad.setEnabled(False)

        layout_derecho.addWidget(label_espi)
        layout_derecho.addWidget(self.btn_tomar_inicio)
        layout_derecho.addWidget(self.spin_cantidad)
        layout_derecho.addWidget(self.spin_velocidad)
        layout_derecho.addWidget(self.btn_tomar_secuencia)

        # Espaciador en la parte inferior del panel derecho
        # Esto "empuja" todos los botones hacia arriba para que no floten en medio de la pantalla
        espaciador = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        layout_derecho.addItem(espaciador)

        # --- ENSAMBLAR TODO ---
        # Al añadir el label de la imagen le damos stretch=1 para que ocupe todo el espacio sobrante
        layout_principal.addWidget(self.label_imagen, stretch=1)
        # Al añadir el panel derecho le damos stretch=0 para que mantenga un tamaño compacto
        layout_principal.addLayout(layout_derecho, stretch=0)

    def _conectar_eventos(self):
        self.btn_iniciar.clicked.connect(self.toggle_camara)
        self.btn_resta.clicked.connect(self.toggle_resta)
        self.btn_filtro.clicked.connect(self.toggle_filtro)
        self.btn_tomar_inicio.clicked.connect(self.capturar_inicio)
        self.btn_tomar_secuencia.clicked.connect(self.iniciar_secuencia)
        self.spin_velocidad.valueChanged.connect(self.actualizar_velocidad)
        
        # NUEVO: Conectar el movimiento del slider
        self.slider_filtro.valueChanged.connect(self.actualizar_etiqueta_filtro)

    # NUEVO: Función para actualizar el texto visual
    def actualizar_etiqueta_filtro(self, valor):
        self.label_radio.setText(f"Radio de Corte: {valor} px")

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
            self.slider_filtro.setEnabled(True) # NUEVO
            
            self.timer_video.start(self.spin_velocidad.value())
        else:
            self.timer_video.stop()
            self.capturando = False
            self.camara.__exit__(None, None, None)
            self.btn_iniciar.setText("Encender Cámara")
            
            # Deshabilitar botones
            self.btn_resta.setEnabled(False)
            self.btn_filtro.setEnabled(False)
            self.slider_filtro.setEnabled(False)
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
        
        # 2. Procesamiento Matemático
        # Hacemos los cálculos ANTES de guardar para capturar el resultado
        imagen_procesada = frame_crudo.astype(np.float32)
        filtro_aplicado = False # Bandera para saber si guardamos la extra

        if self.estado_resta and self.holograma_referencia is not None:
            # Ecuación de ESPI: Valor absoluto de (Actual - Referencia)
            resta = imagen_procesada - self.holograma_referencia
            imagen_procesada = np.abs(resta)
            filtro_aplicado = True

        if self.estado_filtro:
            # NUEVO: Lee el valor exacto en el que pusiste el slider
            radio_actual = self.slider_filtro.value()
            imagen_procesada = filtro_pasa_bajas_frecuencia(imagen_procesada, radio_corte=radio_actual)
            filtro_aplicado = True

        # 3. Lógica de Guardado en Ráfaga
        if self.guardando_secuencia:
            self.contador_toma += 1
            
            # A) Guardar SIEMPRE la toma cruda
            nombre_toma = f"toma_{self.contador_inicio:03d}_{self.contador_toma:03d}.tif"
            ruta_toma = os.path.join(self.carpeta_destino, nombre_toma)
            cv2.imwrite(ruta_toma, self.matriz_cruda_actual)
            
            # B) Guardar ADEMÁS la imagen procesada (si hay un filtro activo)
            if filtro_aplicado:
                nombre_filtrada = f"filtrada_{self.contador_inicio:03d}_{self.contador_toma:03d}.tif"
                ruta_filtrada = os.path.join(self.carpeta_destino, nombre_filtrada)
                # Se guarda en formato TIFF flotante para mantener la máxima precisión
                cv2.imwrite(ruta_filtrada, imagen_procesada)
            
            self.tomas_restantes -= 1
            
            # Detener ráfaga cuando termine
            if self.tomas_restantes <= 0:
                self.guardando_secuencia = False
                self.btn_tomar_inicio.setEnabled(True)
                self.btn_tomar_secuencia.setEnabled(True)
                self.spin_cantidad.setEnabled(True)
                print(f"Secuencia terminada. Archivos guardados en: '{self.carpeta_destino}'")

        # 4. Visualización en la Interfaz
        self.mostrar_imagen(imagen_procesada)

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