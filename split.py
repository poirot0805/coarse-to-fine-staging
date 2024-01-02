import os
rootdir=r"/home/mjy/teeth/datasets/teeth10k/"
bvh_folder=[os.path.join(rootdir,"test","complete"),os.path.join(rootdir,"test","incomplete")]
#[os.path.join(rootdir,"train","complete"),os.path.join(rootdir,"train","incomplete"),os.path.join(rootdir,"val","complete"),os.path.join(rootdir,"val","incomplete")]
total_files=[]
for dataset_path in bvh_folder:
    for f in os.listdir(dataset_path):

        if f.endswith(".json"):
            basename,_=os.path.splitext(f)
            total_files.append(basename+"-1.npy")
with open('/home/mjy/teeth/datasets/teeth10k/test.txt', 'w') as file1:
    for item in total_files:
        file1.write(item + "\n")
# import random

# # 按照8:1的比例随机分配
# random.shuffle(total_files)
# split_index = 8000

# # 分为两个部分
# part1 = total_files[:split_index]
# part2 = total_files[split_index:]

# # 写入两个文本文件
# with open('/home/mjy/teeth/datasets/teeth10k/train.txt', 'w') as file1:
#     for item in part1:
#         file1.write(item + "\n")

# with open('/home/mjy/teeth/datasets/teeth10k/val.txt', 'w') as file2:
#     for item in part2:
#         file2.write(item + "\n")

