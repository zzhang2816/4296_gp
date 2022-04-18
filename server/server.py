import socket
from threading import Thread
from multiprocessing.dummy import Pool as ThreadPool
import zipfile
import os
from time import time

TCP_IP = 'localhost'
TCP_PORT = 9001
BUFFER_SIZE = 1024


class ClientThread:

    def __init__(self, ip, port, sock):
        Thread.__init__(self)
        self.ip = ip
        self.port = port
        self.sock = sock
        print(" New thread started for " + ip + ":" + str(port))

    def run(self, thred_idx):
        print("start", thred_idx)
        self.zip_and_transfer(thred_idx)

        server_reply = self.sock.recv(1024).decode('utf-8')
        print(server_reply)

        self.receive_and_unzip()

        os.remove("tmp.zip")
        self.sock.close()
        print("finish", thred_idx)

    def zip_and_transfer(self, idx):
        filesize = str(os.path.getsize(f'tmp_{idx}.zip'))
        self.sock.sendall(filesize.encode('utf-8'))
        f = open(f'tmp_{idx}.zip', 'rb')
        l = f.read()
        self.sock.sendall(l)

    def receive_and_unzip(self):
        filesize = int(self.sock.recv(1024).decode())
        print(f"receive message from {str(address)}")
        filename = "tmp.zip"
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

THREADS = 0


def run(x):
    sock = x[0]
    thread_idx = x[1]
    sock.run(thread_idx)


THREAD_IDX = 0
while True:
    tcpsock.listen(5)
    print("Waiting for incoming connections...")
    conn, address = tcpsock.accept()
    print('Got connection from ', address)
    newthread = ClientThread(address[0], address[1], conn)
    threads.append([newthread, THREAD_IDX])
    THREAD_IDX += 1
    if len(threads) == 2:
        for i in range(len(threads)):
            with zipfile.ZipFile(f'tmp_{i}.zip', 'w') as file:
                all_files = os.listdir('images')
                batch_size = int(len(all_files) / len(threads))

                if i != len(threads) - 1:
                    files = all_files[batch_size * i: batch_size * (i + 1)]
                else:
                    files = all_files[batch_size * i:]

                for fn in files:
                    file.write(os.path.join('images', fn))

        start_time = time()

        pool = ThreadPool(2)
        pool.map(run, threads)

        print("total computing time %.3f s", time() - start_time)
        for i in range(len(threads)):
            os.remove(f'tmp_{i}.zip')
        break



