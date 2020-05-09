# for循环斐波那契数列
a, b = 0, 1
for num in range(1, 101):
    a, b = b, a + b
    print(f'{num}: {a}')

