from functools import lru_cache

# 斐波那契数列，递归方法
@lru_cache()# 保留函数最近执行的缓存，这里也可以初始化函数的时候定义一个字典存放计算的值
def fib(num):
    if num in (1, 2):
        return 1
    return fib(num - 1) + fib(num - 2)


for num in range(1, 101):
    print(f'{num}: {fib(num)}')
