# Importamos las librerías necesarias
from sympy.strategies.core import switch
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

# Dimensiones del área de texto
ancho_texto = 640
alto_texto = 480  # Similar a la altura del frame para mantener proporciones

conteo = 0
timer = 10
ready = True
detections = False

# Nombre único para la ventana
window_name = "DETECCIÓN Y SEGMENTACIÓN"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # Crear una ventana persistente

# Bucle principal
while True:
    # Leer nuestros fotogramas
    ret, frame = cap.read()
    if not ret:
        break

    # Redimensionar el frame para mantener un formato uniforme
    frame = cv2.resize(frame, (640, 480))

    # Realizar detección con YOLO
    resultados = model(frame, imgsz=256, conf=0.50)

    # Obtener anotaciones para la cámara
    anotaciones = resultados[0].plot()

    # Crear un área para el texto
    texto_area = np.zeros((alto_texto, ancho_texto, 3), dtype=np.uint8)
    imgExample = np.zeros((alto_texto, ancho_texto, 3), dtype=np.uint8)

    # Información a mostrar (por ejemplo, los objetos detectados y su confianza)
    info_text = []
    info_text.append(f"Modelo Entrenado para Deteccion de Latas encendido")
    info_text.append(f"Presione la tecla ESC para terminar proceso")
    conteoNormalizado = conteo * 10000
    if(conteoNormalizado < 10 and ready):
        if (detections):
            estado = "Lata detectada"
            conteo = conteo + .00001
        else:
            estado = "Buscando latas..."
        timer = 10
    else:
        ready = False
        if(timer > 0):
            timer = timer - .1
            estado = f"Buscando deposito... {timer}s."
        else:
            estado = "Depositando latas..."
            conteo = conteo - .00001
            if (conteo < 0):
                ready = True

    info_text.append(f"Latas recolectadas: {conteoNormalizado:.0f}")
    info_text.append(f"Estado de MEDL: {estado}")
    info_text.append(f"Latas detectadas:")
    indice = 0
    detections = False
    for box in resultados[0].boxes:
        indice = indice + 1
        label = int(box.cls.item())  # Convertir índice a entero
        confidence = box.conf.item()  # Convertir confianza a flotante
        info_text.append(f"{indice}: Objeto: {label}, Confianza: {confidence:.2f}")
        detections = True



    # Escribir la información en el área de texto
    y_offset = 20
    for i, line in enumerate(info_text):
        if y_offset + i * line_spacing >= alto_texto - 10:
            break  # Evitar sobreescribir fuera del área de texto
        cv2.putText(texto_area, line, (10, y_offset + i * line_spacing), font, font_scale, (255, 255, 255), font_thickness)

    # Concatenar la vista de cámara con el área de texto horizontalmente
    canvas = np.hstack((anotaciones, texto_area))

    # Mostrar el lienzo en la ventana persistente
    cv2.imshow(window_name, canvas)

    # Cerrar nuestro programa con la tecla ESC
    if cv2.waitKey(1) == 27:
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
