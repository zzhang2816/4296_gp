import socket
from threading import Thread
import zipfile
import os
from time import time

TCP_IP = 'localhost'
TCP_PORT = 9001
BUFFER_SIZE = 1024


class ClientThread(Thread):

    def __init__(self, ip, port, sock):
        Thread.__init__(self)
        self.ip = ip
        self.port = port
        self.sock = sock
        print(" New thread started for " + ip + ":" + str(port))

    def run(self, thred_idx, length):
        self.zip_and_transfer(thred_idx, length)

        server_reply = self.sock.recv(1024).decode('utf-8')
        print(server_reply)

        self.receive_and_unzip()

        os.remove("tmp.zip")
        self.sock.close()

    def zip_and_transfer(self, idx, leng, path='images', zip_name='tmp.zip'):
        with zipfile.ZipFile(zip_name, 'w') as file:
            all_files = os.listdir(path)
            batch_size = int(len(all_files) / leng)

            if idx != leng - 1:
                files = all_files[batch_size * idx: batch_size * (idx + 1)]
            else:
                files = all_files[batch_size * idx:]

            for fn in files:
                file.write(os.path.join(path, fn))
        filesize = str(os.path.getsize('tmp.zip'))
        self.sock.sendall(filesize.encode('utf-8'))
        f = open('tmp.zip', 'rb')
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

while True:
    tcpsock.listen(5)
    print("Waiting for incoming connections...")
    conn, address = tcpsock.accept()
    print('Got connection from ', address)
    newthread = ClientThread(address[0], address[1], conn)
    threads.append(newthread)
    if len(threads) == 2:
        start_time = time()
        for i, thread in enumerate(threads):
            thread.run(i, len(threads))

        print("total computing time %.3f s", time() - start_time)
        break

# for t in threads:
#     t.join()


