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
    cv2.imshow('DETECCION Y SEGMENTACION', anotaciones)

    # Cerrar nuestro programa
    if cv2.waitKey(1) == 27:
        break



cap.release()
cv2.destroyAllWindows()

###############################################################################################

#from ultralytics import YOLO
#import cv2
#import matplotlib.pyplot as plt
#model = YOLO('beta.pt')
#img_path = 'IMG20240904112441.jpg'
#img = cv2.imread(img_path)
#results = model(img)
#annotated_img = results[0].plot()
#plt.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
#plt.axis('off')
#plt.show()
#cv2.imwrite('Results/a.jpg', annotated_img)
