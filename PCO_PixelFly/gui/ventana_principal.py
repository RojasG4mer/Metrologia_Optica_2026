from PyQt5 import QtWidgets, QtCore, QtGui
import numpy as np
import cv2
import os
# from core.pco_camera import CamaraSimulada # O la cámara que estés usando
from core.camara_web import CamaraWebUSB
from procesamiento.operaciones_basicas import normalizar_float_a_uint8
from procesamiento.filtros_en_frecuencia import filtro_pasa_bajas_frecuencia
from datetime import datetime

# Importa la clase que acabas de crear (ajusta la ruta según tu estructura)
from procesamiento.Metodos_desenvolvedores_de_fase import ProcesadorFase

# NUEVO: Importamos el simulador desde su archivo propio 'Simulador_franjas.py'
from procesamiento.Simulador_franjas import SimuladorFranjas

# Para simbolos especiales que se muestren
from PIL import Image, ImageDraw, ImageFont


class VisorInteractivo(QtWidgets.QLabel):
    """Visor que detecta dos clics para dibujar un rectángulo de selección."""
    # Ahora la señal envía 4 datos: x1, y1, x2, y2
    corte_seleccionado = QtCore.pyqtSignal(int, int, int, int)

    def __init__(self, text=""):
        super().__init__(text)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setStyleSheet("background-color: #1e1e1e; color: white; border: 1px solid gray;")
        self.setMinimumSize(640, 480)
        self.imagen_base = None
        self.pt1 = None
        self.pt2 = None

    def set_imagen(self, matriz_uint8):
        self.imagen_base = matriz_uint8.copy()
        self.pt1 = None
        self.pt2 = None
        self.actualizar_visor()

    def actualizar_visor(self):
        if self.imagen_base is None:
            return
            
        img_mostrar = cv2.cvtColor(self.imagen_base, cv2.COLOR_GRAY2RGB)

        # Si solo hay un clic, dibujamos un punto de inicio
        if self.pt1 is not None and self.pt2 is None:
            cv2.circle(img_mostrar, self.pt1, 3, (0, 255, 0), -1)
            
        # Si ya están los dos clics, dibujamos el rectángulo
        elif self.pt1 is not None and self.pt2 is not None:
            cv2.rectangle(img_mostrar, self.pt1, self.pt2, (0, 255, 0), 2)

        alto, ancho, _ = img_mostrar.shape
        q_img = QtGui.QImage(img_mostrar.data, ancho, alto, ancho * 3, QtGui.QImage.Format_RGB888)
        pixmap_escalado = QtGui.QPixmap.fromImage(q_img).scaled(
            self.width(), self.height(), QtCore.Qt.KeepAspectRatio)
        self.setPixmap(pixmap_escalado)

    def mousePressEvent(self, event):
        if self.imagen_base is not None and event.button() == QtCore.Qt.LeftButton:
            pixmap = self.pixmap()
            if not pixmap: return

            offset_x = (self.width() - pixmap.width()) // 2
            offset_y = (self.height() - pixmap.height()) // 2
            click_x = event.x() - offset_x
            click_y = event.y() - offset_y

            if 0 <= click_x <= pixmap.width() and 0 <= click_y <= pixmap.height():
                alto_real, ancho_real = self.imagen_base.shape[:2]
                real_x = int(click_x * ancho_real / pixmap.width())
                real_y = int(click_y * alto_real / pixmap.height())
                
                # Lógica de los dos clics
                if self.pt1 is None or (self.pt1 is not None and self.pt2 is not None):
                    # Primer clic (o reinicio)
                    self.pt1 = (real_x, real_y)
                    self.pt2 = None
                    self.window().btn_procesar.setEnabled(False) # Apagamos botón si reinician el cuadro
                else:
                    # Segundo clic: ordenamos las coordenadas para que siempre sean (arriba-izq) y (abajo-der)
                    x1 = min(self.pt1[0], real_x)
                    y1 = min(self.pt1[1], real_y)
                    x2 = max(self.pt1[0], real_x)
                    y2 = max(self.pt1[1], real_y)
                    
                    self.pt1 = (x1, y1)
                    self.pt2 = (x2, y2)
                    self.corte_seleccionado.emit(x1, y1, x2, y2)
                    
                self.actualizar_visor()


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
        # os.makedirs(self.carpeta_destino, exist_ok=True)
        
        # Variables de procesamiento
        self.estado_resta = False
        self.estado_filtro = False
        
        # Temporizador para el bucle de video
        self.timer_video = QtCore.QTimer()
        self.timer_video.timeout.connect(self.adquirir_y_procesar)

        self._configurar_ui()
        self._conectar_eventos()

    def _configurar_ui(self):
            """Este método SOLO crea las pestañas y llama a los constructores"""
            self.tabs = QtWidgets.QTabWidget()
            self.setCentralWidget(self.tabs)
            
            # Creamos los 3 contenedores vacíos
            self.tab_adquisicion = QtWidgets.QWidget()
            self.tab_analisis = QtWidgets.QWidget()
            self.tab_simulador = QtWidgets.QWidget()
            
            # Los agregamos al menú superior
            self.tabs.addTab(self.tab_adquisicion, "📷 1. Adquisición ESPI")
            self.tabs.addTab(self.tab_analisis, "📊 2. Análisis y Fase")
            self.tabs.addTab(self.tab_simulador, "🌊 3. Simulador")
            
            # LLAMAMOS A LOS 3 CONSTRUCTORES (Esto evita tu error)
            self._construir_ui_adquisicion()
            self._construir_ui_analisis()
            self._construir_ui_simulador()

    def _construir_ui_adquisicion(self):
        """Este es TU código exacto de la cámara, pero adaptado a su pestaña"""
        # 1. Layout Principal Horizontal conectado a la pestaña 1
        layout_principal = QtWidgets.QHBoxLayout(self.tab_adquisicion)

        # --- PANEL IZQUIERDO (VISUALIZACIÓN) ---
        self.label_imagen = QtWidgets.QLabel("Esperando conexión...")
        self.label_imagen.setAlignment(QtCore.Qt.AlignCenter)
        self.label_imagen.setStyleSheet("background-color: black; color: white; border: 1px solid gray;")
        self.label_imagen.setMinimumSize(640, 480) 

        # --- PANEL DERECHO (CONTROLES) ---
        layout_derecho = QtWidgets.QVBoxLayout()
        layout_derecho.setContentsMargins(15, 10, 15, 10) 
        layout_derecho.setSpacing(10) 

        # Sección: Hardware y Procesamiento en Vivo
        label_hardware = QtWidgets.QLabel("<b>Control de Cámara y Filtros</b>")
        label_hardware.setAlignment(QtCore.Qt.AlignCenter)
        
        self.btn_iniciar = QtWidgets.QPushButton("Encender Cámara")
        self.btn_resta = QtWidgets.QPushButton("Ver Resta (ESPI): OFF")
        self.btn_filtro = QtWidgets.QPushButton("Ver Pasa Bajas: OFF")
        
        # Control dinámico para el tamaño del filtro
        self.label_radio = QtWidgets.QLabel("Radio de Corte: 80 px")
        self.label_radio.setAlignment(QtCore.Qt.AlignCenter)
        
        self.slider_filtro = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_filtro.setRange(1, 500)   
        self.slider_filtro.setValue(80)       
        
        # Control dinámico para el Enfoque
        self.label_enfoque = QtWidgets.QLabel("Enfoque (Solo WebCam): Auto")
        self.label_enfoque.setAlignment(QtCore.Qt.AlignCenter)
        
        self.slider_enfoque = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_enfoque.setRange(0, 255) 
        self.slider_enfoque.setValue(128)    
        
        # Deshabilitar botones y sliders al inicio
        self.btn_resta.setEnabled(False)
        self.btn_filtro.setEnabled(False)
        self.slider_filtro.setEnabled(False) 
        self.slider_enfoque.setEnabled(False) 

        layout_derecho.addWidget(label_hardware)
        layout_derecho.addWidget(self.btn_iniciar)
        layout_derecho.addWidget(self.btn_resta)
        layout_derecho.addWidget(self.btn_filtro)
        layout_derecho.addWidget(self.label_radio)   
        layout_derecho.addWidget(self.slider_filtro) 
        layout_derecho.addWidget(self.label_enfoque)  
        layout_derecho.addWidget(self.slider_enfoque) 
    
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
        espaciador = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        layout_derecho.addItem(espaciador)

        # --- ENSAMBLAR TODO ---
        layout_principal.addWidget(self.label_imagen, stretch=1)
        layout_principal.addLayout(layout_derecho, stretch=0)

    def _construir_ui_simulador(self):
        layout_principal = QtWidgets.QHBoxLayout(self.tab_simulador)
        
        # --- PANEL IZQUIERDO: CUADRÍCULA DE VISORES (GRID) ---
        layout_visores = QtWidgets.QGridLayout()
        
        # 1. Sección Superior: Franjas Originales
        label_titulo_top = QtWidgets.QLabel("<b>Franjas Originales</b>")
        label_titulo_top.setAlignment(QtCore.Qt.AlignCenter)
        
        self.label_sim_top = QtWidgets.QLabel("Genera una simulación...")
        self.label_sim_top.setAlignment(QtCore.Qt.AlignCenter)
        self.label_sim_top.setStyleSheet("background-color: #1e1e1e; color: white; border: 1px solid gray;")
        self.label_sim_top.setMinimumSize(450, 300)
        
        # 2. Sección Inferior Izquierda: Fase Envuelta
        label_titulo_env = QtWidgets.QLabel("<b>Fase Envuelta (-π a π)</b>")
        label_titulo_env.setAlignment(QtCore.Qt.AlignCenter)
        
        self.label_sim_env = QtWidgets.QLabel("Esperando procesamiento...")
        self.label_sim_env.setAlignment(QtCore.Qt.AlignCenter)
        self.label_sim_env.setStyleSheet("background-color: #1e1e1e; color: white; border: 1px solid gray;")
        self.label_sim_env.setMinimumSize(220, 220)
        
        # 3. Sección Inferior Derecha: Fase Desenvuelta
        label_titulo_desenv = QtWidgets.QLabel("<b>Fase Desenvuelta (Continua)</b>")
        label_titulo_desenv.setAlignment(QtCore.Qt.AlignCenter)
        
        self.label_sim_desenv = QtWidgets.QLabel("Esperando procesamiento...")
        self.label_sim_desenv.setAlignment(QtCore.Qt.AlignCenter)
        self.label_sim_desenv.setStyleSheet("background-color: #1e1e1e; color: white; border: 1px solid gray;")
        self.label_sim_desenv.setMinimumSize(220, 220)
        
        # --- DISTRIBUCIÓN EN LA CUADRÍCULA (GRID) ---
        # Fila 0: Título superior
        layout_visores.addWidget(label_titulo_top, 0, 0, 1, 2)
        # Fila 1: Visor superior (ocupa 2 columnas)
        layout_visores.addWidget(self.label_sim_top, 1, 0, 1, 2)
        
        # Fila 2: Títulos inferiores
        layout_visores.addWidget(label_titulo_env, 2, 0)
        layout_visores.addWidget(label_titulo_desenv, 2, 1)
        
        # Fila 3: Visores inferiores lado a lado
        layout_visores.addWidget(self.label_sim_env, 3, 0)
        layout_visores.addWidget(self.label_sim_desenv, 3, 1)
        
        # --- PANEL DERECHO: CONTROLES ---
        layout_derecho = QtWidgets.QVBoxLayout()
        layout_derecho.setContentsMargins(15, 10, 15, 10)
        layout_derecho.setSpacing(10)
        
        label_titulo = QtWidgets.QLabel("<b>Simulación y Validación</b>")
        label_titulo.setAlignment(QtCore.Qt.AlignCenter)
        
        self.combo_tipo_franja = QtWidgets.QComboBox()
        self.combo_tipo_franja.addItems([
            "Lineales (Tilt)", 
            "Circulares (Desenfocamiento)",
            "Chirp Exponencial"
        ])
        
        self.combo_metodo_sim = QtWidgets.QComboBox()
        self.combo_metodo_sim.addItems(["3 Pasos", "4 Pasos", "Carré (4 Pasos)", "Hariharan (5 Pasos)"])
        
        self.btn_generar = QtWidgets.QPushButton("⚙️ Simular y Procesar")
        self.btn_guardar_sim = QtWidgets.QPushButton("💾 Guardar Stack (.tif)")
        self.btn_guardar_sim.setEnabled(False)
        
        # Ensamblaje del panel de controles
        layout_derecho.addWidget(label_titulo)
        layout_derecho.addWidget(QtWidgets.QLabel("Patrón Geométrico:"))
        layout_derecho.addWidget(self.combo_tipo_franja)
        layout_derecho.addWidget(QtWidgets.QLabel("Método Phase Shifting:"))
        layout_derecho.addWidget(self.combo_metodo_sim)
        
        linea = QtWidgets.QFrame()
        linea.setFrameShape(QtWidgets.QFrame.HLine)
        linea.setFrameShadow(QtWidgets.QFrame.Sunken)
        layout_derecho.addWidget(linea)
        
        layout_derecho.addWidget(self.btn_generar)
        layout_derecho.addWidget(self.btn_guardar_sim)
        
        espaciador = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        layout_derecho.addItem(espaciador)
        
        # --- ENSAMBLAR TODO EN LA PESTAÑA ---
        layout_principal.addLayout(layout_visores, stretch=1)
        layout_principal.addLayout(layout_derecho, stretch=0)
        
        self.simulador_core = SimuladorFranjas(resolucion=512)
        self.stack_simulado = None

    def _conectar_eventos(self):
        self.btn_iniciar.clicked.connect(self.toggle_camara)
        self.btn_resta.clicked.connect(self.toggle_resta)
        self.btn_filtro.clicked.connect(self.toggle_filtro)
        self.btn_tomar_inicio.clicked.connect(self.capturar_inicio)
        self.btn_tomar_secuencia.clicked.connect(self.iniciar_secuencia)
        self.spin_velocidad.valueChanged.connect(self.actualizar_velocidad)
        # (Debajo de tus conexiones anteriores...)
        self.btn_cargar_imgs.clicked.connect(self.cargar_multiples_imagenes)
        self.btn_procesar.clicked.connect(self.ejecutar_extraccion_y_desenvolvimiento)
        self.btn_guardar_fase.clicked.connect(self.guardar_mapa_fase)
        
        # Al cambiar de método, limpiamos la memoria para evitar errores
        self.combo_metodo.currentIndexChanged.connect(self.limpiar_memoria_imagenes)
        # NUEVO: Conectar el movimiento del slider
        self.slider_filtro.valueChanged.connect(self.actualizar_etiqueta_filtro)
        
        # NUEVO: Conectar el movimiento del slider de enfoque
        self.slider_enfoque.valueChanged.connect(self.actualizar_enfoque)

        # Conexión del nuevo visor (rectángulo)
        self.label_analisis.corte_seleccionado.connect(self.registrar_corte_fourier)
        self.btn_generar.clicked.connect(self.ejecutar_simulacion)

        # Conexiones del Simulador
        self.btn_generar.clicked.connect(self.ejecutar_simulacion)
        
        # ---> AGREGA ESTA LÍNEA <---
        self.btn_guardar_sim.clicked.connect(self.guardar_stack_simulado)


    # NUEVO: Función para actualizar el texto visual
    def actualizar_etiqueta_filtro(self, valor):
        self.label_radio.setText(f"Radio de Corte: {valor} px")

    def actualizar_enfoque(self, valor):
        self.label_enfoque.setText(f"Enfoque (Solo WebCam): {valor}")
        
        # Solo envía el comando si la cámara actual tiene el método 'set_focus' programado
        if self.camara and hasattr(self.camara, 'set_focus'):
            self.camara.set_focus(valor)

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
            self.slider_filtro.setEnabled(True) # Filtro
            self.slider_enfoque.setEnabled(True) # Enfoque
            
            self.timer_video.start(self.spin_velocidad.value())
        else:
            self.timer_video.stop()
            self.capturando = False
            self.camara.__exit__(None, None, None)
            self.btn_iniciar.setText("Encender Cámara")
            
            # Deshabilitar botones
            self.btn_resta.setEnabled(False)
            self.btn_filtro.setEnabled(False)
            self.slider_filtro.setEnabled(False) # Filtro
            self.slider_enfoque.setEnabled(False) # Enfoque

            
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

    def _asegurar_carpeta_existe(self):
        """Crea la carpeta de destino solo si aún no existe."""
        if not os.path.exists(self.carpeta_destino):
            os.makedirs(self.carpeta_destino)
            print(f"Carpeta de sesión creada: {self.carpeta_destino}")
    # --- LÓGICA DE GUARDADO ESPECÍFICA ---
    def capturar_inicio(self):
        """Guarda la imagen base y prepara la secuencia."""
        if hasattr(self, 'imagen_actual') and self.imagen_actual is not None:
            # NUEVO: Verificamos/creamos la carpeta justo antes de guardar
            self._asegurar_carpeta_existe() 
            
            ruta = os.path.join(self.carpeta_destino, "inicio_001.tif")
            cv2.imwrite(ruta, self.imagen_actual)
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
        
        self._asegurar_carpeta_existe()
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

    def mostrar_imagen_en_label(self, img_uint8, label_destino):
        """Muestra una imagen matriz uint8 en un QLabel específico."""
        alto, ancho = img_uint8.shape
        q_img = QtGui.QImage(img_uint8.data, ancho, alto, ancho, QtGui.QImage.Format_Grayscale8)
        pixmap = QtGui.QPixmap.fromImage(q_img).scaled(
            label_destino.width(), label_destino.height(), QtCore.Qt.KeepAspectRatio)
        label_destino.setPixmap(pixmap)


    def _construir_ui_analisis(self):
            layout_principal = QtWidgets.QHBoxLayout(self.tab_analisis)
        
            # --- PANEL IZQUIERDO: NUEVO VISOR INTERACTIVO ---
            self.label_analisis = VisorInteractivo("Selecciona un método y carga las imágenes...")
            
            # --- PANEL DERECHO: Controles de Fase ---
            layout_derecho = QtWidgets.QVBoxLayout()
            layout_derecho.setContentsMargins(15, 10, 15, 10)
            layout_derecho.setSpacing(10)
            
            label_titulo = QtWidgets.QLabel("<b>Extracción y Desenvolvimiento</b>")
            label_titulo.setAlignment(QtCore.Qt.AlignCenter)
            



            # 1. Selector de Método
            self.combo_metodo = QtWidgets.QComboBox()
            self.combo_metodo.addItems([
                "Fourier (Requiere 2 imágenes)",
                "Phase Shifting - 3 Pasos (Requiere 3)",
                "Phase Shifting - 4 Pasos (Requiere 4)",
                "Phase Shifting - Carré (Requiere 4)",
                "Phase Shifting - Hariharan (Requiere 5)"
            ])
            
            # 2. Controles de Carga
            # Debe decir btn_cargar_imgs (con 's')
            self.btn_cargar_imgs = QtWidgets.QPushButton("📂 Cargar Imágenes (.tif)")
            self.label_estado_carga = QtWidgets.QLabel("Imágenes en memoria: 0")
            self.label_estado_carga.setStyleSheet("color: #aaaaaa;")
            
            # 3. Controles de Procesamiento y Guardado
            self.btn_procesar = QtWidgets.QPushButton("🗺️ Calcular y Desenvolver Fase")
            self.btn_guardar_fase = QtWidgets.QPushButton("💾 Guardar Mapa de Fase")
            
            self.btn_procesar.setEnabled(False)
            self.btn_guardar_fase.setEnabled(False)
            
            # Ensamblaje del panel derecho
            layout_derecho.addWidget(label_titulo)
            layout_derecho.addWidget(QtWidgets.QLabel("Método de Extracción:"))
            layout_derecho.addWidget(self.combo_metodo)
            
            linea1 = QtWidgets.QFrame()
            linea1.setFrameShape(QtWidgets.QFrame.HLine)
            linea1.setFrameShadow(QtWidgets.QFrame.Sunken)
            layout_derecho.addWidget(linea1)
            
            layout_derecho.addWidget(self.btn_cargar_imgs)
            layout_derecho.addWidget(self.label_estado_carga)
            
            linea2 = QtWidgets.QFrame()
            linea2.setFrameShape(QtWidgets.QFrame.HLine)
            linea2.setFrameShadow(QtWidgets.QFrame.Sunken)
            layout_derecho.addWidget(linea2)
            
            layout_derecho.addWidget(self.btn_procesar)
            layout_derecho.addWidget(self.btn_guardar_fase)
            
            espaciador = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
            layout_derecho.addItem(espaciador)
            
            # Ensamblar pestaña completa
            layout_principal.addWidget(self.label_analisis, stretch=1)
            layout_principal.addLayout(layout_derecho, stretch=0)
            
            # Instanciar el procesador matemático
            self.procesador_fase = ProcesadorFase()
            self.stack_imagenes = []
            self.mapa_fase_desenvuelta = None

    # =========================================================================
    # LÓGICA DE LA PESTAÑA DE ANÁLISIS DE FASE
    # =========================================================================

    def limpiar_memoria_imagenes(self):
        """Reinicia el estado al cambiar de método en el menú desplegable."""
        self.stack_imagenes = []
        self.label_estado_carga.setText("Imágenes en memoria: 0")
        self.btn_procesar.setEnabled(False)
        self.btn_guardar_fase.setEnabled(False)

    def cargar_multiples_imagenes(self):
            """Abre un diálogo que permite seleccionar múltiples archivos a la vez."""
            opciones = QtWidgets.QFileDialog.Options() # Agregamos QtWidgets.
            rutas_archivos, _ = QtWidgets.QFileDialog.getOpenFileNames( # Agregamos QtWidgets.
                self,
                "Seleccionar Imágenes de Franjas (Selección múltiple)",
                getattr(self, 'carpeta_destino', ""), 
                "Imágenes TIFF (*.tif *.tiff);;Todos los archivos (*)",
                options=opciones
            )
            
            if rutas_archivos:
                self.stack_imagenes = []
                for ruta in sorted(rutas_archivos): 
                    img = cv2.imread(ruta, cv2.IMREAD_UNCHANGED)
                    if img is not None:
                        self.stack_imagenes.append(img.astype(np.float32))
                
                self.label_estado_carga.setText(f"Imágenes en memoria: {len(self.stack_imagenes)}")
                self.btn_procesar.setEnabled(True)
            # (Después del for loop donde cargas las imágenes)
            if self.stack_imagenes and "Fourier" in self.combo_metodo.currentText():
                # CAMBIO AQUÍ: Ahora pedimos exactamente 2 imágenes
                if len(self.stack_imagenes) == 2:
                    # Usamos la primera imagen (Referencia) para buscar el espectro
                    self.F_actual, espectro_log = self.procesador_fase.obtener_espectro_fourier(self.stack_imagenes[0])
                    
                    e_min, e_max = np.min(espectro_log), np.max(espectro_log)
                    espectro_vis = ((espectro_log - e_min) / (e_max - e_min + 1e-8) * 255).astype(np.uint8)
                    
                    self.label_analisis.set_imagen(espectro_vis)
                    self.fourier_box = None
                    self.btn_procesar.setEnabled(False) 
                    QtWidgets.QMessageBox.information(self, "Fourier", "Haz 2 clics para enmarcar el lóbulo lateral.")
                else:
                    QtWidgets.QMessageBox.warning(self, "Fourier", "Para este método debes seleccionar exactamente 2 imágenes: Referencia y Deformada.")

    def ejecutar_extraccion_y_desenvolvimiento(self):
        """Valida la cantidad de imágenes y ejecuta el método matemático elegido."""
        if not self.stack_imagenes:
            return

        metodo_seleccionado = self.combo_metodo.currentText()
        cantidad_cargada = len(self.stack_imagenes)
        
        try:
            # 1. Extracción de la fase envuelta según el método
            if "Fourier" in metodo_seleccionado:
                if not hasattr(self, 'fourier_box') or self.fourier_box is None:
                    raise ValueError("Debes hacer 2 clics para dibujar un rectángulo sobre el lóbulo antes de desenvolver.")
                
                x1, y1, x2, y2 = self.fourier_box
                
                # NUEVA LLAMADA: Le pasamos imagen 0 (Ref), imagen 1 (Def) y las 4 coordenadas
                fase_envuelta = self.procesador_fase.fase_fourier_dos_imagenes(
                    self.stack_imagenes[0], 
                    self.stack_imagenes[1], 
                    x1, y1, x2, y2
                )
                
            elif "3 Pasos" in metodo_seleccionado:
                if cantidad_cargada != 3:
                    raise ValueError(f"Cargaste {cantidad_cargada} imágenes, pero se requieren 3.")
                fase_envuelta = self.procesador_fase.phase_shifting_3_pasos(self.stack_imagenes)
                
            elif "4 Pasos" in metodo_seleccionado:
                if cantidad_cargada != 4:
                    raise ValueError(f"Cargaste {cantidad_cargada} imágenes, pero se requieren 4.")
                fase_envuelta = self.procesador_fase.phase_shifting_4_pasos(self.stack_imagenes)
                
            elif "Carré" in metodo_seleccionado:
                if cantidad_cargada != 4:
                    raise ValueError(f"Cargaste {cantidad_cargada} imágenes, pero se requieren 4.")
                fase_envuelta = self.procesador_fase.phase_shifting_carre(self.stack_imagenes)
                
            elif "Hariharan" in metodo_seleccionado:
                if cantidad_cargada != 5:
                    raise ValueError(f"Cargaste {cantidad_cargada} imágenes, pero se requieren 5.")
                fase_envuelta = self.procesador_fase.phase_shifting_5_pasos_hariharan(self.stack_imagenes)

            # 2. Desenvolvimiento (Unwrapping)
            # Primero intenta el método robusto, si falla, usa el básico
            fase_desenvuelta = self.procesador_fase.desenvolvimiento_robusto(fase_envuelta)
            if fase_desenvuelta is None:
                fase_desenvuelta = self.procesador_fase.desenvolvimiento_basico(fase_envuelta)

            # Guardar el mapa de fase final en memoria para exportación
            self.mapa_fase_desenvuelta = fase_desenvuelta
            
            # 3. Mostrar en pantalla (Requiere mapear de radianes flotantes a 0-255 uint8)
            fase_min, fase_max = np.min(fase_desenvuelta), np.max(fase_desenvuelta)
            fase_visual = (fase_desenvuelta - fase_min) / (fase_max - fase_min + 1e-8) * 255
            self.mostrar_imagen_analisis(fase_visual.astype(np.uint8))
            
            self.btn_guardar_fase.setEnabled(True)
            print("Extracción y desenvolvimiento completados con éxito.")

        # Desenvolvimiento y visualización
            fase_desenvuelta = self.procesador_fase.desenvolvimiento_robusto(fase_envuelta)
            if fase_desenvuelta is None:
                fase_desenvuelta = self.procesador_fase.desenvolvimiento_basico(fase_envuelta)

            self.mapa_fase_desenvuelta = fase_desenvuelta
            
            fase_min, fase_max = np.min(fase_desenvuelta), np.max(fase_desenvuelta)
            fase_visual = ((fase_desenvuelta - fase_min) / (fase_max - fase_min + 1e-8) * 255).astype(np.uint8)
            
            # En lugar de usar tu mostrar_imagen_analisis anterior, usamos el nuevo:
            self.label_analisis.set_imagen(fase_visual)
            
            self.btn_guardar_fase.setEnabled(True)

        except ValueError as ve:
            QtWidgets.QMessageBox.warning(self, "Error de Validación", str(ve))
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error Matemático", f"Ocurrió un error en el procesamiento:\n{e}")

    def mostrar_imagen_analisis(self, img_uint8):
        """Renderiza una matriz uint8 de 0-255 en el visor de la Pestaña 2."""
        alto, ancho = img_uint8.shape
        q_img = QtGui.QImage(img_uint8.data, ancho, alto, ancho, QtGui.QImage.Format_Grayscale8)
        self.label_analisis.setPixmap(QtGui.QPixmap.fromImage(q_img).scaled(
            self.label_analisis.width(), self.label_analisis.height(), QtCore.Qt.KeepAspectRatio))

    def guardar_mapa_fase(self):
        """Exporta la fase desenvuelta cruda en formato TIFF de 32 bits."""
        if self.mapa_fase_desenvuelta is None:
            return
            
        opciones = QtWidgets.QFileDialog.Options() # Agregamos QtWidgets.
        ruta_archivo, _ = QtWidgets.QFileDialog.getSaveFileName( # Agregamos QtWidgets.
            self,
            "Guardar Fase Desenvuelta",
            "fase_desenvuelta",
            "Mapa de Fase TIFF (*.tif *.tiff)",
            options=opciones
        )
        
        if ruta_archivo:
            if not ruta_archivo.lower().endswith(('.tif', '.tiff')):
                ruta_archivo += '.tif'
                
            cv2.imwrite(ruta_archivo, self.mapa_fase_desenvuelta.astype(np.float32))
            print(f"Fase guardada exitosamente en: {ruta_archivo}")
            QtWidgets.QMessageBox.information(self, "Guardado Exitoso", "El mapa de fase se ha exportado correctamente.")

    def guardar_stack_simulado(self):
        if not hasattr(self, 'ultima_panoramica'):
            return
            
        opciones = QtWidgets.QFileDialog.Options()
        ruta_archivo, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Guardar Reporte Visual",
            "reporte_simulacion",
            "Imagen PNG (*.png);;Imagen JPEG (*.jpg)",
            options=opciones
        )
        
        if ruta_archivo:
            # 1. Generamos la imagen compuesta
            reporte_final = self.generar_reporte_visual(
                self.ultima_panoramica, 
                self.ultima_envuelta, 
                self.ultima_desenvuelta
            )
            
            # 2. Asegurar extensión y guardar
            if not ruta_archivo.lower().endswith(('.png', '.jpg', '.jpeg')):
                ruta_archivo += '.png'
                
            cv2.imwrite(ruta_archivo, reporte_final)
            QtWidgets.QMessageBox.information(self, "Éxito", f"Reporte gráfico guardado correctamente en:\n{ruta_archivo}")

    def actualizar_radio(self, valor):
        """Actualiza en tiempo real el tamaño del círculo de corte."""
        self.label_analisis.radio = valor
        self.label_analisis.actualizar_visor()

    def registrar_clic_fourier(self, x, y):
        """Guarda las coordenadas y permite desenvolver la fase."""
        self.fourier_cx = x
        self.fourier_cy = y
        self.btn_procesar.setEnabled(True)

    def registrar_corte_fourier(self, x1, y1, x2, y2):
        """Guarda las coordenadas del rectángulo y habilita el cálculo."""
        self.fourier_box = (x1, y1, x2, y2)
        self.btn_procesar.setEnabled(True)

    def ejecutar_simulacion(self):
        tipo_texto = self.combo_tipo_franja.currentText().lower()
        metodo_texto = self.combo_metodo_sim.currentText()
        
        # 1. Determinar número de pasos basado en el método
        if "3 Pasos" in metodo_texto: pasos = 3
        elif "Hariharan" in metodo_texto: pasos = 5
        else: pasos = 4  # 4 Pasos clásico o Carré
            
        if "lineales" in tipo_texto: tipo_fase = 'lineal'
        elif "circulares" in tipo_texto: tipo_fase = 'circular'
        elif "espiral" in tipo_texto: tipo_fase = 'espiral'
        elif "chirp" in tipo_texto: tipo_fase = 'chirp'
            
        try:
            # --- FASE 1: GENERACIÓN DEL STACK ---
            self.stack_simulado, _ = self.simulador_core.generar_stack_phase_shifting(
                tipo_fase=tipo_fase, pasos=pasos)
            
            # --- MOSTRAR TODAS LAS IMÁGENES CON UN ESPACIO SEPARADOR ---
            stack_uint8 = [(img * 255).astype(np.uint8) for img in self.stack_simulado]
            
            # 1. Creamos una barra separadora vertical negra de 4 píxeles de ancho
            alto, ancho = stack_uint8[0].shape
            separador = np.zeros((alto, 50), dtype=np.uint8) # Fondo negro de separación
            
            # 2. Intercalamos las imágenes con el separador
            stack_con_separacion = []
            for i, img in enumerate(stack_uint8):
                stack_con_separacion.append(img)
                if i < len(stack_uint8) - 1:  # Evita poner separador después de la última imagen
                    stack_con_separacion.append(separador)
            
            # 3. Unimos todo en una sola vista panorámica con espacios divisorios claros
            self.ultima_panoramica = np.hstack(stack_con_separacion)
            
            # La mostramos en el visor superior
            self.mostrar_imagen_en_label(self.ultima_panoramica, self.label_sim_top)
            
            # --- FASE 2: EXTRACCIÓN (FASE ENVUELTA) ---
            if "3 Pasos" in metodo_texto:
                fase_env = self.procesador_fase.phase_shifting_3_pasos(self.stack_simulado)
            elif "Carré" in metodo_texto:
                fase_env = self.procesador_fase.phase_shifting_carre(self.stack_simulado)
            elif "Hariharan" in metodo_texto:
                fase_env = self.procesador_fase.phase_shifting_5_pasos_hariharan(self.stack_simulado)
            else:
                fase_env = self.procesador_fase.phase_shifting_4_pasos(self.stack_simulado)
                
            self.ultima_envuelta = ((fase_env + np.pi) / (2 * np.pi) * 255).astype(np.uint8)
            self.mostrar_imagen_en_label(self.ultima_envuelta, self.label_sim_env)
            
            # --- FASE 3: DESENVOLVIMIENTO ---
            fase_desenv = self.procesador_fase.desenvolvimiento_robusto(fase_env)
            if fase_desenv is None:
                fase_desenv = self.procesador_fase.desenvolvimiento_basico(fase_env)
                
            f_min, f_max = np.min(fase_desenv), np.max(fase_desenv)
            self.ultima_desenvuelta = ((fase_desenv - f_min) / (f_max - f_min + 1e-8) * 255).astype(np.uint8)
            self.mostrar_imagen_en_label(self.ultima_desenvuelta, self.label_sim_desenv)
            
            self.btn_guardar_sim.setEnabled(True)
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Fallo al simular: {e}")



    def generar_reporte_visual(self, img_superior, img_izq, img_der, 
                               titulo_sup="Franjas Originales", 
                               titulo_izq="Fase Envuelta (-π a π)", 
                               titulo_der="Fase Desenvuelta (Continua)"):
        """
        Construye un canvas de alta resolución con OpenCV y Pillow 
        para renderizar correctamente símbolos Unicode (como π).
        """
        # 1. Convertir a BGR para el fondo
        if len(img_superior.shape) == 2: img_superior = cv2.cvtColor(img_superior, cv2.COLOR_GRAY2BGR)
        if len(img_izq.shape) == 2: img_izq = cv2.cvtColor(img_izq, cv2.COLOR_GRAY2BGR)
        if len(img_der.shape) == 2: img_der = cv2.cvtColor(img_der, cv2.COLOR_GRAY2BGR)

        h_sup, w_sup = img_superior.shape[:2]
        h_izq, w_izq = img_izq.shape[:2]
        h_der, w_der = img_der.shape[:2]

        espacio_x = 20
        margen = 40
        espacio_y_titulos = 35

        ancho_inf = w_izq + espacio_x + w_der
        ancho_lienzo = max(w_sup, ancho_inf) + (margen * 2)
        alto_lienzo = margen + espacio_y_titulos + h_sup + margen + espacio_y_titulos + max(h_izq, h_der) + margen

        # 2. Crear el fondo oscuro con NumPy
        lienzo = np.full((alto_lienzo, ancho_lienzo, 3), 34, dtype=np.uint8)

        # 3. Colocar las imágenes y dibujar bordes con OpenCV
        y_txt_sup = margen
        y_img_sup = y_txt_sup + 20
        y_txt_inf = y_img_sup + h_sup + 40
        y_img_inf = y_txt_inf + 20

        x_sup = (ancho_lienzo - w_sup) // 2
        lienzo[y_img_sup:y_img_sup+h_sup, x_sup:x_sup+w_sup] = img_superior

        x_izq = (ancho_lienzo - ancho_inf) // 2
        x_der = x_izq + w_izq + espacio_x

        lienzo[y_img_inf:y_img_inf+h_izq, x_izq:x_izq+w_izq] = img_izq
        lienzo[y_img_inf:y_img_inf+h_der, x_der:x_der+w_der] = img_der

        color_borde = (100, 100, 100)
        cv2.rectangle(lienzo, (x_sup-1, y_img_sup-1), (x_sup+w_sup, y_img_sup+h_sup), color_borde, 1)
        cv2.rectangle(lienzo, (x_izq-1, y_img_inf-1), (x_izq+w_izq, y_img_inf+h_izq), color_borde, 1)
        cv2.rectangle(lienzo, (x_der-1, y_img_inf-1), (x_der+w_der, y_img_inf+h_der), color_borde, 1)

        # ==========================================
        # 4. RENDERIZADO DE TEXTO CON PILLOW
        # ==========================================
        # Convertir de OpenCV (BGR) a Pillow (RGB)
        lienzo_pil = Image.fromarray(cv2.cvtColor(lienzo, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(lienzo_pil)

        # Cargar fuente cruzada (Windows/Mac)
        try:
            fuente = ImageFont.truetype("arial.ttf", 22)
        except IOError:
            try:
                fuente = ImageFont.truetype("Arial.ttf", 22) # Formato común en Mac
            except IOError:
                fuente = ImageFont.load_default()

        def poner_texto_pil(texto, y, ancho_total, x_offset=0):
            # Calcular dimensiones del texto para centrarlo
            caja = draw.textbbox((0, 0), texto, font=fuente)
            ancho_texto = caja[2] - caja[0]
            x = x_offset + (ancho_total - ancho_texto) // 2
            # Dibujar en blanco puro
            draw.text((x, y), texto, font=fuente, fill=(255, 255, 255))

        # Escribir los títulos
        poner_texto_pil(titulo_sup, y_txt_sup - 20, ancho_lienzo)
        poner_texto_pil(titulo_izq, y_txt_inf - 20, w_izq, x_izq)
        poner_texto_pil(titulo_der, y_txt_inf - 20, w_der, x_der)

        # 5. Regresar el lienzo al formato de OpenCV (BGR) para el guardado
        lienzo_final = cv2.cvtColor(np.array(lienzo_pil), cv2.COLOR_RGB2BGR)
        return lienzo_final