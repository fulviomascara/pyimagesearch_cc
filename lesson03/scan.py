# importar bibliotecas necessárias
from pyimagesearch.transform import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils

# construir o parser de argumentos e realizar o parse
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
    help = "caminho da imagem a ser escaneada")
args = vars(ap.parse_args())

# carregar a imagem, calcular o ratio da altura antiga
# para a altura nova, clonar a imagem e redimensionar
# o ratio é calculado para que o scan seja feito na imagem original
# e não na imagem redimensionada
image = cv2.imread(args["image"])
ratio = image.shape[0] / 500.0
orig = image.copy()
image = imutils.resize(image, height=500)

# converter a imagem pra Grayscale, aplicar Blur e
# localizar bordas
# TO DO - Verificar como localizar bordas nas minhas notas fiscais
# ajustando os parâmetros do Canny Edge Detection ou verificando
# outro tipo de abordagem
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 75, 200)

# mostrar a imagem original e a imagem com as bordas detectadas
print("[INFO] Passo 1 - Detecção de Bordas")
cv2.imshow("Image", image)
cv2.imshow("Bordas Detectadas", edged)
cv2.waitKey(0)
cv2.destroyAllWindows()

# localizar os contornos da imagem com as bordas detectadas,
# mantendo apenas as maiores, e inicialize o contorno na tela
# a premissa assumida aqui é que um scanner digitaliza um pedaço
# de papel, que é um retângulo e em geral é o maior retângulo (externo)
# detectado na imagem
# a abordagem abaixo retorna uma lista de contornos e ordena pela área
# da maior para a menor, mantendo apenas as 5 maiores
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key = cv2.contourArea, reverse=True)[:5]

# loop nos contornos
for c in cnts:
    # aproximar os Contornos
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    # se o contorno aproximado tem 4 pontos,
    # vamos assumir que encontramos nosso maior retângulo
    if len(approx) == 4:
        screenCnt = approx
        break

# mostrar o contorno (borda) em volta do papel a ser escaneado
print("[INFO] Passo 2 - Encontrar os contornos do Papel")
cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
cv2.imshow("Contornos", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# aplicar uma transformação de quatro pontos para obter
# uma visão top-down da imagem original
# TO-DO ... entender a função de warp transform feita pelo
# Adrian Rosebrock na função four_point_transform
warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

# converte a imagem para Grayscale, aplica threshold
# coloque efeito 'preto e branco' de Papel
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
T = threshold_local(warped, 11, offset = 10, method = "gaussian")
warped = (warped > T).astype("uint8") * 255

# mostrar imagem original e imagem escaneada
print("[INFO] Passo 3 - Aplicar transformação de perspectiva")
cv2.imshow("Original", imutils.resize(orig, height = 650))
cv2.imshow("Escaneada", imutils.resize(warped, height = 650))
cv2.waitKey(0)
cv2.destroyAllWindows()
