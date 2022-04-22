import socket
from threading import Thread
import zipfile
import os
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-ip', type=str, help='ip address', default="127.0.0.1")
args = parser.parse_args()

TCP_IP = args.ip
TCP_PORT = 9999
BUFFER_SIZE = 1024


class ClientThread(Thread):

    def __init__(self, ip, port, sock, idx):
        Thread.__init__(self)
        self.ip = ip
        self.port = port
        self.sock = sock
        self.idx = idx
        print(" New thread started for " + ip + ":" + str(port))

    def run(self):
        start_time = time.time()
        print('start thread', self.idx)
        self.zip_and_transfer()

        server_reply = self.sock.recv(1024).decode('utf-8')
        print(server_reply)

        self.receive_and_unzip()

        os.remove(f"tmp_{self.idx}.zip")

        self.sock.close()

        print('finish thread', self.idx, 'in', time.time() - start_time, 's')

    def zip_and_transfer(self):
        filesize = str(os.path.getsize(f'tmp_{self.idx}.zip'))
        self.sock.sendall(filesize.encode('utf-8'))
        f = open(f'tmp_{self.idx}.zip', 'rb')
        l = f.read()
        self.sock.sendall(l)

    def receive_and_unzip(self):
        filesize = int(self.sock.recv(1024).decode())
        print(f"receive message from {str(address)}")
        filename = f"tmp_{self.idx}.zip"
        f = open(filename, 'wb')
        client_data = self.sock.recv(1024)
        total = 0
        while client_data:
            f.write(client_data)
            total += len(client_data)
            if total == filesize:
                break
            client_data = self.sock.recv(1024)
        f.close()
        with zipfile.ZipFile(filename, 'r') as file:
            print('[+] Extracting files...')
            file.extractall()
            print('[+] Done')


tcpsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
tcpsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
tcpsock.bind((TCP_IP, TCP_PORT))
threads = []

THREAD_IDX = 0
while True:
    tcpsock.listen(5)
    print("Waiting for incoming connections...")
    conn, address = tcpsock.accept()
    print('Got connection from ', address)
    newthread = ClientThread(address[0], address[1], conn, THREAD_IDX)
    THREAD_IDX += 1
    threads.append(newthread)
    if len(threads) == 2:

        for i in range(len(threads)):
            with zipfile.ZipFile(f'tmp_{i}.zip', 'w') as file:
                all_files = os.listdir('images')
                batch_size = int(len(all_files) / len(threads))

                if THREAD_IDX != len(threads) - 1:
                    files = all_files[batch_size * i: batch_size * (i + 1)]
                else:
                    files = all_files[batch_size * i:]

                for fn in files:
                    file.write(os.path.join('images', fn))

        start_time = time.time()
        for i, thread in enumerate(threads):
            thread.start()

        break

for t in threads:
    t.join()

print("total computing time:", time.time() - start_time, 's')
