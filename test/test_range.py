def chunk_list(lst, chunk_size):
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

my_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
for batch in chunk_list(my_list, 12):
    print(batch)  # [1, 2, 3], [4, 5, 6], [7, 8, 9]
r = range(0, 9, 3)
print(r)
test_list = []
for batch_num, i in enumerate(range(0, len(test_list), 3), 1):
    print(f"批次 {batch_num}: {my_list[i:i + 3]}")


my_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
# 可以这样使用 enumerate 包装 chunk_list
for batch_num, batch in enumerate(chunk_list(my_list, 3), 1):
    print(f"批次 {batch_num}: {batch}")
