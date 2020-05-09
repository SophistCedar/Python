from socket import socket,gethostname


def main():
    # 1.创建套接字对象默认使用IPv4和TCP协议
    client = socket()    
    host = "192.168.14.128" # 获取本地主机名
    port = 1234        # 设置端口
    # 2.连接到服务器(需要指定IP地址和端口)
    client.connect((host, port))    
    # 3.从服务器接收数据
    running = True
    while running:
        print(client.recv(1024).decode('utf-8'))
    client.close()


if __name__ == '__main__':
    main()
