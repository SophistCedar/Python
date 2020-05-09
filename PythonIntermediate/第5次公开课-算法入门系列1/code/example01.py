import time
nums = []
start = time.time()
for i in range(100000):
    nums.append(i)
nums.reverse()
end = time.time()
print(nums)
print(f'花费时间：{end-start}')
