import sys
from PyQt5 import QtWidgets
from gui.ventana_principal import VentanaPrincipal

def iniciar_aplicacion():
    # Inicializa el motor de la interfaz gráfica
    app = QtWidgets.QApplication(sys.argv)
    
    # Crea y muestra la ventana principal
    ventana = VentanaPrincipal()
    ventana.show()
    
    # Mantiene la aplicación corriendo hasta que la cierres
    sys.exit(app.exec_())

if __name__ == "__main__":
    iniciar_aplicacion()