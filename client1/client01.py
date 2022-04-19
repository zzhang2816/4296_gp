import socket
import zipfile
import os
import shutil
import torch
from time import  time

TCP_IP = 'localhost'
TCP_PORT = 9001
BUFFER_SIZE = 1024


def zip_and_transfer(path='images', zip_name='tmp.zip'):
    with zipfile.ZipFile(zip_name, 'w') as file:
        for fn in os.listdir(path):
            file.write(os.path.join(path, fn))
    filesize = str(os.path.getsize('tmp.zip'))
    s.sendall(str(filesize).encode())
    f = open('tmp.zip', 'rb')
    l = f.read()
    s.sendall(l)


def receive_and_unzip():
    filesize = int(s.recv(1024).decode())
    filename = "tmp.zip"
    f = open(filename, 'wb')
    client_data = s.recv(1024)
    total = 0

    while client_data:
        f.write(client_data)
        total += len(client_data)
        if total == filesize:
            break
        client_data = s.recv(1024)
    f.close()

    with zipfile.ZipFile(filename, 'r') as file:
        print('[+] Extracting files...')
        file.extractall()
        print('[+] Done')


s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((TCP_IP, TCP_PORT))

receive_and_unzip()

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

imgs = os.listdir('images')  # batch of images
imgs = list(map(lambda img: os.path.join('images/', img), imgs))

# Inference
results = model(imgs)

# Results
results.save()  # or .show()

s.sendall('inference finished'.encode('utf-8'))

# 4. reply the prediction results
zip_and_transfer(path="runs/detect/exp")

s.close()

os.remove("tmp.zip")
shutil.rmtree("images")
shutil.rmtree("runs")

