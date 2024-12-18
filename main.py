# Importamos las librerias
from ultralytics import YOLO
import cv2

# Leer nuestro modelo
model = YOLO("epsilon.pt")

# Realizar VideoCaptura
cap = cv2.VideoCapture(0)

# Bucle
while True:
    # Leer nuestros fotogramas
    ret, frame = cap.read()

    # Leemos resultados
    resultados = model(frame, imgsz = 256, conf = 0.50)

    # Mostramos resultados
    anotaciones = resultados[0].plot()


    # Mostramos nuestros fotogramas
    #cv2.namedWindow('DETECCION Y SEGMENTACION', cv2.WND_PROP_FULLSCREEN)
    #cv2.setWindowProperty('DETECCION Y SEGMENTACION', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    #cv2.imshow('DETECCION Y SEGMENTACION', anotaciones)

    # Cerrar nuestro programa
    if cv2.waitKey(1) == 27:
        break



cap.release()
#cv2.destroyAllWindows()