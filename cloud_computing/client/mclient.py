import socket
import zipfile
import os
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('-ip', type=str, help='ip address', default="127.0.0.1")
args = parser.parse_args()

ip_port = (args.ip, 9999)
s = socket.socket()
s.connect(ip_port)


def zip_and_transfer(path='images', zip_name='tmp.zip'):
    with zipfile.ZipFile(zip_name, 'w') as file:
        for fn in os.listdir(path):
            file.write(os.path.join(path, fn))
    filesize = str(os.path.getsize('tmp.zip'))
    f = open('tmp.zip', 'rb')
    l = f.read()
    s.sendall(l)


def receive_and_unzip():
    filename = "tmp.zip"
    f = open(filename, 'wb')
    client_data = s.recv(1024)
    total = 0
    while (client_data):
        f.write(client_data)
        total += len(client_data)
        client_data = s.recv(1024)
    f.close()
    with zipfile.ZipFile(filename, 'r') as file:
        print('[+] Extracting files...')
        file.extractall()
        print('[+] Done')


start_time = time.time()
zip_and_transfer()
s.close()

s = socket.socket()
s.connect(ip_port)
server_reply = s.recv(1024).decode('utf-8')
print(server_reply)

receive_and_unzip()
print("total time: ", time.time() - start_time)
os.remove("tmp.zip")
s.close()
