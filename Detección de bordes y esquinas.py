import numpy as np
import cv2


# Cargamos la imagen
imgB = cv2.imread('BORDES.jpg')
cv2.imshow("original", imgB)

# Convertimos a escala de grises
gris = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)
 
# Aplicar suavizado Gaussiano
gauss = cv2.GaussianBlur(gris, (5,5), 0)
 
cv2.imshow("suavizado", gauss)

# Detectamos los bordes con Canny
canny = cv2.Canny(gauss, 50, 150)
 
cv2.imshow("canny", canny)

# Buscamos los contornos
(contornos,_) = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Mostramos el número de monedas por consola
print("EL programa a encontrado {} objetos".format(len(contornos)))

cv2.drawContours(imgB,contornos,-1,(0,0,255), 2)
cv2.imshow("contornos", imgB)

cv2.waitKey(0)

imgE = cv2.imread('ESQUINAS.jpg')
gray = cv2.cvtColor(imgE,cv2.COLOR_BGR2GRAY)

corners = cv2.goodFeaturesToTrack(gray, 40, 0.01, 10)
corners = np.int0(corners)
   
for i in corners:
    x,y = i.ravel()
    cv2.circle(imgE, (x,y), 3, 255, -1)
   
cv2.imshow('Detector de Esquinas', imgE)
cv2.waitKey(0)