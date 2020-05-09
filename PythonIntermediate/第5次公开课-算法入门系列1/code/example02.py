import time 
nums = []
start = time.time()
for i in range(100000):
    nums.insert(0, i)
print(nums)
end = time.time()
print(f'花费时间：{end-start}') 
#此处花费时间较多是因为，列表的底层存储是连续的，每次插入一个元素，会把这个元素后面的元素全部后移一位