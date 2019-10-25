# importar bibliotecas necessárias
import argparse
import cv2
import imutils

# construir o Parse de Argumentos e fazer o Parse
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
    help='caminho da imagem de entrada')
args = vars(ap.parse_args())

# carregar e apresentar a imagem de entrada
image = cv2.imread(args['image'])
cv2.imshow("Imagem", image)
cv2.waitKey(0)

# converter a imagem para greyscale
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Imagem Grayscale", image_gray)
cv2.waitKey(0)

# aplicar detecção de bordas nos objetos da imagem,
# usando o algoritmo Canny (John F. Canny - 1986)
# minThreshold, maxThreshold, Sobel kernel size=3
image_edged = cv2.Canny(image_gray, 30, 150)
cv2.imshow("Bordas", image_edged)
cv2.waitKey(0)

# aplicar threshold, colocando todos os pixels < 225 em branco (frente)
# e pixels >= 225 em preto (fundo)
# usando a imagem em Grayscale
image_threshold = cv2.threshold(image_gray, 225, 255, cv2.THRESH_BINARY_INV)[1]
cv2.imshow("Threshold", image_threshold)
cv2.waitKey(0)

# encontrar os contornos dos objetos, com base na imagem com threshold
# RETR_EXTERNAL = apenas os contornos mais externos
# CHAIN_APPROX_SIMPLE = comprime os segmentos verticais, horizontais e diagonais,
# mantendo só os pontos 'finais'
cnts = cv2.findContours(image_threshold.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
output = image.copy()

for c in cnts:
    # desenhar um contorno em torno de cada objeto encontrado
    # com base na imagem original
    cv2.drawContours(output, [c], -1, (255, 200, 0), 3)
    cv2.imshow('Contornos', output)
    cv2.waitKey(0)

# colocar um texto com a quantidade de objetos encontrados
text = "Encontrei {} objetos".format(len(cnts))
cv2.putText(output, text, (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
cv2.imshow('Contornos', output)
cv2.waitKey(0)

# reduzir contorno (erosion) com base na image Threshold
image_mask = image_threshold.copy()
image_mask = cv2.erode(image_mask, None, iterations=5)
cv2.imshow('Imagem com Redução de Contorno', image_mask)
cv2.waitKey(0)

# dilatar contorno (dllate) com base na image Threshold
image_mask = image_threshold.copy()
image_mask = cv2.dilate(image_mask, None, iterations=5)
cv2.imshow('Imagem com Dilatação de Contorno', image_mask)
cv2.waitKey(0)

# aplicar 'mask' através de uma operação de bitwise
# neste caso, faremos uma combinação da imagem threshold (fundo preto)
# com a imagem original (peças do Tetris em Cores)
image_mask = image_threshold.copy()
output = cv2.bitwise_and(image, image, mask=image_mask)
cv2.imshow("Imagem com Mask", output)
cv2.waitKey(0)
