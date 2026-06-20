import cv2

class CamaraWebUSB:
    def __init__(self, indice_camara=1):
        """
        En Mac, el índice 0 suele ser la cámara FaceTime integrada.
        El índice 1 normalmente es la primera cámara USB externa conectada.
        """
        self.camera_name = "Microsoft LifeCam HD-6000"
        self.indice = indice_camara
        self.cap = None

    def __enter__(self):
        print(f"[{self.camera_name}] Abriendo puerto USB...")
        self.cap = cv2.VideoCapture(self.indice)
        
        if not self.cap.isOpened():
            raise Exception(f"No se detectó cámara en el índice {self.indice}.")
            
        # Intentamos forzar la resolución nativa HD de la LifeCam (720p)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cap:
            self.cap.release()
            print(f"[{self.camera_name}] Puerto USB liberado.")

    def record(self, number_of_images=1, mode='sequence'):
        # Como OpenCV lee el flujo constante por USB, no hacemos nada aquí
        pass 

    def image(self):
        ret, frame = self.cap.read()
        if not ret:
            raise Exception("Error al leer el frame de la cámara web.")
            
        # Convertimos la imagen a escala de grises (1 canal) 
        # para que tus filtros de metrología óptica funcionen correctamente
        frame_gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        alto, ancho = frame_gris.shape
        
        return frame_gris, {'width': ancho, 'height': alto, 'camera': self.camera_name}