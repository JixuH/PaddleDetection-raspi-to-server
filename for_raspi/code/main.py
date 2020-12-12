import _thread
import socket
import subprocess
import time

from watchdog.events import *
from watchdog.observers import Observer

from FileSystemEvent import MyDirEventHandler

# 执行预测
def run_detection(sh_name):
    subprocess.run(sh_name,shell=True)
# 监控文件的变化、发送文件至终端
def run_scan_file(file_name):
    # 创建观察者对象
    observer = Observer()
    # 创建事件处理对象
    fileHandler = MyDirEventHandler()
    # 为观察者设置观察对象与处理事件对象
    observer.schedule(
        fileHandler, file_name, True)
    observer.start()
    print(fileHandler)
    try:
        while True:
            time.sleep(2)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


"""
多线程
run_detection 执行预测程序
run_scan_file 监控文件的变化、发送文件至终端
"""
if __name__ == '__main__':
    try:
    	# 预测程序
        _thread.start_new_thread(run_detection,('./real_time.sh',))
        # 监控地址（绝对路径）
        _thread.start_new_thread(run_scan_file,("/home/pi/Desktop/PaddleDetection-raspi-to-server(tcp)/for_raspi/code/screenshots",))
    except:
        print('无法启动线程')
