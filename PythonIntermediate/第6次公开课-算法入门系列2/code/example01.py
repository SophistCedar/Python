import sys


def fac(num):
    if num == 0:
        return 1
    return num * fac(num - 1)


def main():
    print(fac(1200))


if __name__ == '__main__':
    sys.setrecursionlimit(60000) # 更改迭代的深度
    main()
# for i in range(1000):
#     print(f'{i}:'.rjust(3), fac(i))

