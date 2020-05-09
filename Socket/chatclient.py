from socket import socket,gethostname
from threading import Thread


def main():

    class RefreshScreenThread(Thread):

        def __init__(self, client):
            super().__init__()
            self._client = client

        def run(self):
            while running:
                data = self._client.recv(1024)
                print(data.decode('utf-8'))

    nickname = input('请输入你的昵称: ')
    myclient = socket()
    host = gethostname() # 获取本地主机名
    myclient.connect((host, 12345))
    running = True
    RefreshScreenThread(myclient).start()
    while running:
        content = input('请发言: ')
        if content == 'byebye':
            myclient.send(content.encode('utf-8'))
            running = False
        else:
            msg = nickname + ': ' + content
            myclient.send(msg.encode('utf-8'))


if __name__ == '__main__':
    main()
