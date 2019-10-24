# importar bibliotecas necessárias
import cv2
import imutils

# carregue a imagem de input e mostre suas dimensões, tendo em mente que
# as imagens são representadas em um array NumPy multidimensional com o shape
# numero de linhas (altura) X numero de colunas (largura) X numero de canais (profundidade)
image = cv2.imread('images/jp.jpeg')
(h, w, d) = image.shape
print("largura={}, altura={}, profundidade={}".format(w, h, d))

# accesse o pixel RGB localizado na posição X=50, y=100, tendo em mente que
# o OpenCV guarda as imagens no formato BGR ao invés de RGB
(B, G, R) = image[100, 50]
print("R={}, G={}, B={}".format(R, G, B))

# mostre a imagem na tela
# clique na janela ou pressione uma tecla pra continuar a execução
cv2.imshow("Imagem", image)
cv2.waitKey(0)

# extrair um retângulo 124x167 para um ROI (Região de Interesse) da Imagem
# de entrada, começando na posição x=27, y=24 e terminando em x=151, y=191
# slice de imagem [startY:endY, startX:endX]
roi = image[24:191, 27:151]
cv2.imshow("ROI", roi)
cv2.waitKey(0)

# redimensionar a imagem para 300x300, ignorando o "aspect ratio"
resized = cv2.resize(image, (300,300))
cv2.imshow("Redimensionamento Fixo", resized)
cv2.waitKey(0)

# redimensionar a imagem para 300 de largura, mas calculando a altura
# baseando-se no "aspect ratio"
ratio = 300.0 / w
dim = (300, int(h * ratio))
resized_ratio = cv2.resize(image, dim)
cv2.imshow("Redimensionamento com Ratio", resized_ratio)
cv2.waitKey(0)

# redimensionar a imagem para 200 de altura, usando a biblioteca imutils
# do Adrian Rosebrock
resized_easy = imutils.resize(image, height=200)
cv2.imshow("Redimensionamento com imutils", resized_easy)
cv2.waitKey(0)

# rotacionar uma imagem em 45 graus (sentido horário), calculando o centro da imagem,
# construir uma matriz de rotação e finalmente aplicação o "warp affine"
center = (w // 2, h // 2)
rotation_matrix = cv2.getRotationMatrix2D(center, -45, 1.0)
rotated = cv2.warpAffine(image, rotation_matrix, (w, h))
cv2.imshow("Rotação da imagem, com Rotation Matrix e Warp Affine", rotated)
cv2.waitKey(0)

# rotacionar uma imagem em 30 graus (sentido horário), usando a biblioteca imutils
# do Adrian Rosebrock
rotated_easy = imutils.rotate(image, -30)
cv2.imshow("Rotação da imagem, com imutils", rotated_easy)
cv2.waitKey(0)

# rotacionar uma imagem em 45 graus (sentido horário), usando a biblioteca imutils
# do Adrian Rosebrock, sem o cliping da mesma
# Nesta função, o nível da rotação (em graus) deve ser o inverso (negativos - sentido antihorario,
# positivos - sentido horário)
rotated_bound = imutils.rotate_bound(image, 30)
cv2.imshow("Rotação da imagem, com imutils, sem clipping", rotated_bound)
cv2.waitKey(0)

# aplicar Gaussian Blur na imagem, com um kernel de 11x11 para suavizar,
# útil para reduzir "high frequency noise"
blurred = cv2.GaussianBlur(image, (11, 11), 0)
cv2.imshow("Borrada", blurred)
cv2.waitKey(0)

# desenhar um retângulo vermelho em volta da face
# (top, left), (bottom, right), (B, G, R), thickness
output_rectangle = image.copy()
cv2.rectangle(output_rectangle, (27, 24), (151,191), (0, 0, 255), 2)
cv2.imshow("Retângulo", output_rectangle)
cv2.waitKey(0)

# desenhar um retângulo vermelho em volta da face
# (y, x), radius, (R, G, R), thickness
output_circle = image.copy()
cv2.circle(output_circle, (489, 38), 30, (255, 0, 0), -1)
cv2.imshow("Circulo", output_circle)
cv2.waitKey(0)

# desenhar uma linha vermelha na imagem
# (top, left), (bottom, right), (B, G, R), thickness
output_line = image.copy()
cv2.line(output_line, (27, 24), (151,191), (0, 0, 255), 2)
cv2.imshow("Linha", output_line)
cv2.waitKey(0)

# escrever um texto na Imagem
# (top, left), font, scale, (B, G, R), thickness
output_text = image.copy()
cv2.putText(output_text, "OpenCV + Jurassic Park !!!", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)
cv2.imshow("Texto", output_text)
cv2.waitKey(0)
