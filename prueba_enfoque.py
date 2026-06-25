import cv2

# En Mac, la cámara externa suele ser el índice 1 o 2 (el 0 es la integrada)
# Usamos obligatoriamente el backend de Apple (CAP_AVFOUNDATION)
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    print("Índice 1 falló. Probando índice 2...")
    cap = cv2.VideoCapture(2, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    print("Índice 2 falló. Probando índice 0 (Cámara integrada)...")
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    print("Error: No se pudo otorgar acceso a ninguna cámara en macOS.")
    exit()

print("¡Cámara conectada exitosamente en Mac!")

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    cv2.imshow('LifeCam HD-6000 en Mac', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
