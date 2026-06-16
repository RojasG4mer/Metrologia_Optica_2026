import pco

def verificar_hardware():
    print("Buscando PCO PixelFly...")
    try:
        # Intentamos abrir la conexión real
        with pco.Camera() as cam:
            print("========================================")
            print("✅ ¡CONEXIÓN EXITOSA!")
            print(f"📸 Modelo detectado: {cam.camera_name}")
            print(f"🌡️ Temperatura del sensor: {cam.sdk.get_temperature().get('sensor', 'N/A')} °C")
            print("========================================")
    except Exception as e:
        print("\n❌ FALLO EN LA CONEXIÓN")
        print(f"Detalle del error: {e}")
        print("Revisa que PCO Camware esté cerrado (no puedes usar la cámara en dos programas a la vez).")

if __name__ == "__main__":
    verificar_hardware()