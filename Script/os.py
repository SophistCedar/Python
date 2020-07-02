import os

fd = os.open("test.txt",os.O_CREAT|os.O_RDWR)
print(fd)

content = "just a test".encode('utf-')
n = os.write(fd,content)
print(n)

l = os.lseek(fd,0,os.SEEK_SET)
print(l)


content_read = os.read(fd,n)
print(content_read)

os.close(fd)
