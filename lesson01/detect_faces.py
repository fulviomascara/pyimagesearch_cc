# importar bibliotecas
import numpy as np
import argparse
import cv2

# construir o parser de argumentos e fazer o parse
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
    help="caminho para a imagem de entrada")
ap.add_argument("-p", "--prototxt", required=True,
    help="caminho para o arquivo prototxt do Caffe")
ap.add_argument("-m", "--model", required=True,
    help="caminho para o modelo pré-treinado do Caffe")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
    help="probabilidade mínima para filtrar detecções fracas")
args = vars(ap.parse_args())

# carregar o modelo serializado do HD
print("[INFO] carregando modelo...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# carregar a imagem de entrada e construir um blob pra esta imagem
# através do resize para um tamanho fixo de 300x300 pixels e então normalizar
# COMMENT: Por que estes valores para normalização? https://www.pyimagesearch.com/2017/11/06/deep-learning-opencvs-blobfromimage-works/
image = cv2.imread(args["image"])
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
    (300,300), (104.0, 117.0, 123.0))

# passe o blob pela rede neural, obtenha as detecções e
# predições
print("[INFO] Calculando detecções de objetos")
net.setInput(blob)
detections = net.forward()

# loop sob as detecções
for i in range(0, detections.shape[2]):
    # extrair a confiança (probabilidade) associada com a
    # predição
    confidence = detections[0, 0, i, 2]

    # Eliminar detecções fracas, garantindo que a confiança é
    # maior que a confiança esperada (argumento)
    if confidence > args["confidence"]:
        # calcular as coordenadas (X, y) do bounding box para o
        # objeto
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        # desenhar o bounding box do rosto com a probabilidade
        # associada
        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(image, (startX, startY), (endX, endY),
            (0, 0, 255), 2)
        cv2.putText(image, text, (startX, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

# mostrar a imagem de saída
cv2.imshow("Output", image)
cv2.waitKey(0)
