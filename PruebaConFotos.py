from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

modelName = 'epsilon'
model = YOLO(modelName + '.pt')
img_path = 'AI4.jfif'
img = cv2.imread(img_path)
results = model(img)
annotated_img = results[0].plot()

plt.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
cv2.imwrite('Results/' + modelName + img_path, annotated_img)
