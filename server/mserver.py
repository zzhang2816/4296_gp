import socket
import zipfile
from src import detect
import os
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-ip', type=str, help='ip address',default="127.0.0.1")
args = parser.parse_args()


def zip_and_transfer(path = 'images',zip_name = 'tmp.zip'):
    with zipfile.ZipFile(zip_name, 'w') as file:
        for fn in os.listdir(path):
            file.write(os.path.join(path,fn))
    filesize = str(os.path.getsize('tmp.zip'))
    conn.sendall(filesize.encode())
    f = open('tmp.zip', 'rb')
    l = f.read()
    conn.sendall(l)

def receive_and_unzip():
    filesize = int(conn.recv(1024).decode())
    print(f"receive message from {str(address)}" )
    filename = "tmp.zip"
    f = open(filename, 'wb')
    client_data = conn.recv(1024)
    total = 0    
    while(client_data):
        f.write(client_data)
        total += len(client_data)
        if(total == filesize):
            break
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

# 3. start inference
conn.sendall('start inference'.encode())
detect.main("images")


# 4. reply the prediction results
zip_and_transfer(path="labels")

# 5. do the clear and close the connection
os.remove("tmp.zip")
shutil.rmtree("images")
shutil.rmtree("labels")

conn.close()
sk.close()

