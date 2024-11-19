# Importamos las librerías necesarias
from ultralytics import YOLO
import cv2
import numpy as np

# Leer nuestro modelo
model = YOLO("epsilon.pt")

# Realizar VideoCaptura
cap = cv2.VideoCapture(0)

# Configuración de tamaño del texto
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
font_thickness = 1
line_spacing = 15  # Espaciado entre líneas de texto

# Bucle principal
while True:
    # Leer nuestros fotogramas
    ret, frame = cap.read()
    if not ret:
        break

    # Realizar detección con YOLO
    resultados = model(frame, imgsz=256, conf=0.50)

    # Obtener anotaciones para la cámara
    anotaciones = resultados[0].plot()

    # Crear un área para el texto (del mismo ancho que el frame)
    altura_texto = 200
    texto_area = np.zeros((altura_texto, frame.shape[1], 3), dtype=np.uint8)

    # Información a mostrar (por ejemplo, los objetos detectados y su confianza)
    info_text = []
    for box in resultados[0].boxes:
        label = box.cls  # Etiqueta del objeto detectado
        confidence = box.conf  # Confianza del modelo
        info_text.append(f"Objeto: {label}, Confianza: {confidence:.2f}")

    # Escribir la información en el área de texto
    y_offset = 10
    for i, line in enumerate(info_text):
        cv2.putText(texto_area, line, (10, y_offset + i * line_spacing), font, font_scale, (255, 255, 255), font_thickness)

    # Concatenar la vista de cámara con el área de texto
    canvas = np.vstack((anotaciones, texto_area))

    # Mostrar la ventana
    cv2.imshow("DETECCIÓN Y SEGMENTACIÓN", canvas)

    # Cerrar nuestro programa con la tecla ESC
    if cv2.waitKey(1) == 27:
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()

