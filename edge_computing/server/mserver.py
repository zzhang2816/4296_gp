import socket
import zipfile
import os
import shutil
import argparse
import torch
import time

parser = argparse.ArgumentParser()
parser.add_argument('-ip', type=str, help='ip address', default="127.0.0.1")
args = parser.parse_args()


def zip_and_transfer(path='image', zip_name='tmp.zip'):
    with zipfile.ZipFile(zip_name, 'w') as file:
        for fn in os.listdir(path):
            file.write(os.path.join(path, fn))
    filesize = str(os.path.getsize('tmp.zip'))
    print(filesize)
    f = open('tmp.zip', 'rb')
    l = f.read()
    conn.sendall(l)


def receive_and_unzip():
    print(f"receive message from {str(address)}")
    filename = "tmp.zip"
    f = open(filename, 'wb')
    client_data = conn.recv(1024)
    total = 0
    while (client_data):
        f.write(client_data)
        total += len(client_data)
        client_data = conn.recv(1024)
    f.close()
    with zipfile.ZipFile(filename, 'r') as file:
        print('[+] Extracting files...')
        file.extractall()
        print('[+] Done')


# 1. set up socket
ip_port = (args.ip, 9999)
sk = socket.socket()
sk.bind(ip_port)
sk.listen(5)
print('socket service start，wait for client to connect...')
conn, address = sk.accept()

# 2. receive transfer images
receive_and_unzip()
conn.close()

conn, address = sk.accept()
# 3. start inference
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

imgs = os.listdir('images')  # batch of images
imgs = list(map(lambda img: os.path.join('images/', img), imgs))


# Inference
start_time = time.time()
results = model(imgs)
print("total computing time for yolo:", time.time() - start_time, 's')

results.save()
conn.sendall('inference finished'.encode('utf-8'))

# 4. reply the prediction results
zip_and_transfer(path="runs/detect/exp")

# 5. do the clear and close the connection
os.remove("tmp.zip")
shutil.rmtree("images")

conn.close()
sk.close()
