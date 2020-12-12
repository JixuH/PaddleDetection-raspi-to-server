import glob
import os
import socket
import struct
import sys
import time


class Client():
    def __init__(self, file_path):
        self.file_path = file_path

    def sock_client_image(self, file_path):
        while True:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.connect(('106.12.118.56', 6000))
            except socket.error as msg:
                print(msg)
                print(sys.exit(1))

            #按照图片格式、图片名字、图片大小打包
            fhead = struct.pack(b'128sq', 
                                bytes(os.path.basename(self.file_path), encoding='utf-8'),
                                os.stat(self.file_path).st_size) 
            s.send(fhead)
            # 打开要传输的图片
            fp = open(self.file_path, 'rb')  
            while True:
                # 读入图片数据，一次只读取1024个字节
                data = fp.read(1024)  
                if not data:
                    print('{0} send over...'.format(self.file_path))
                    break
                # 以二进制格式发送图片数据（每次发1024个字节）
                s.send(data)
            s.close()
            #time.sleep(10)
            # 删除图片
            #os.remove(file_path)
            break
