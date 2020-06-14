import sys
import numpy as np

class Logger(object):
    def __init__(self,filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename,"a")
        
    def write(self,message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        pass


class Unbuffered(object):
   def __init__(self, stream):
       self.stream = stream
   def write(self, data):
       self.stream.write(data)
       self.stream.flush()
   def writelines(self, datas):
       self.stream.writelines(datas)
       self.stream.flush()
   def __getattr__(self, attr):
       return getattr(self.stream, attr)
       
       
def single_np(arr, target):
    arr = np.array(arr)
    mask = (arr == target)
    arr_new = arr[mask]
    return mask


def deletefiles(dir):
    import os
    files = os.listdir(dir)  #列出目录下的文件
    for file in files:
        os.remove(dir+file)    #删除文件
        print(file + " deleted")
    return


def mkdir(path):
    # 引入模块
    import os
 
    # 去除首位空格
    path=path.strip()
    # 去除尾部 \ 符号
    path=path.rstrip("\\")
 
    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists=os.path.exists(path)
 
    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path) 
 
        print( path+' 创建成功')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print( path+' 目录已存在')
        return False


def my_resize(image,w_des,h_des):
    import cv2
    h, w, ch, = image.shape
    if h/h_des > 1 or w/w_des >1:
        h_n = int(h / max(h / h_des, w / w_des) )
        w_n = int(w / max(h / h_des, w / w_des))
    else: #h <= h_des and w <= w_des
        h_n, w_n = h, w
    # print('h_n = {},w_n = {}'.format(h_n,w_n))
    image_resized = cv2.resize(image, (w_n,h_n))
    return image_resized        
  