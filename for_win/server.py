import os
import socket
import struct
import sys


# TCP接收数据
def deal_image(sock, address):
    # 初始化SQL
    MySQL = MySQL_Connect()
    print("Accept connection from {0}".format(address))  # 查看发送端的ip和端口

    while True:
        fileinfo_size = struct.calcsize('128sq')  # 返回格式字符串fmt描述的结构的字节大小
        print('fileinfo_size is', fileinfo_size)
        buf = sock.recv(fileinfo_size)  # 接收图片名
        print('buf is ', buf)
        if buf: 
            # 解码
            filename, filesize = struct.unpack(
                '128sq', buf)  
            print('filename :', filename.decode(), 'filesize :', filesize)
            fn = filename.decode().strip('\x00')
            print('fn is ', fn)
            # 获取路径
            current_path = os.getcwd().replace('\\','/')
            File_Path = current_path + '/Data_received/' + fn[1:13] + '/'
            # 检查此文件夹是否存在
            if not os.path.exists(File_Path):
                os.makedirs(File_Path)
            # 在服务器端新建图片名
            new_filename = os.path.join(File_Path + fn)
            print(new_filename)

            recvd_size = 0
            # 二进制打开文件
            fp = open(new_filename, 'wb')  
            # 接收数据 每次从客户端接受1024个字节
            while not recvd_size == filesize:
                if filesize - recvd_size > 1024:
                    data = sock.recv(1024)
                    recvd_size += len(data)
                else: 
                    data = sock.recv(1024)
                    recvd_size = filesize
                # 写入图片数据
                fp.write(data)
            fp.close()
        sock.close()
        break

# 开始连接 接收存储数据
def socket_service_image():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(('127.0.0.1', 22))
        s.listen(30)
    except socket.error as msg:
        print(msg)
        sys.exit(1)
    print("Wait for Connection...")
    while True:
        sock, address = s.accept()  # addr是一个元组(ip,port)
        deal_image(sock, address)

if __name__ == '__main__':
    # socket
    socket_service_image()
