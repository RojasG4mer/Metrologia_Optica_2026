def __init__(self):
        super().__init__()
        # ... otras variables ...

        # Solo preparamos el nombre, PERO YA NO LA CREAMOS AQUÍ
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.carpeta_destino = f"capturas_espi_{timestamp}"
        
        # BORRA O COMENTA ESTA LÍNEA:
        # os.makedirs(self.carpeta_destino, exist_ok=True) 
        
        self._configurar_ui()
        self._conectar_eventos()