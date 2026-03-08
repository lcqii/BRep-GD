import numpy as np
from hashlib import sha256
from data_process.convert_utils import real2bit
import pickle
import os
import math
from occwl.io import load_step
from data_process.process_brep import parse_solid
text2int  = {'bathtub':0, 'bed':1, 'bench':2, 'bookshelf':3,'cabinet':4, 'chair':5, 'couch':6, 'lamp':7, 'sofa':8, 'table':9}
text2int_cadnet50 = {'AngleIron': 0, 'Bearing': 1, 'BearingHouse': 2, 'Bolt': 3, 'BoringBar': 4, 'Cross_Pipe': 5, 'Elbow': 6, 'Flange': 7, 'FlatKey': 8, 'ForgedShackle': 9, 
                    'Gear': 10, 'Grease_Seal': 11, 'Grooved_Pin': 12, 'Handwheel': 13, 'HeaveTightCouplingSleeve': 14, 'HookWrench': 15, 'Key': 16, 'KeylessDrillChuck': 17, 'Lathe_Center': 18, 'LiftingHook': 19,
                    'Lock_Washer': 20, 'Nut': 21, 'Pipe_Clip': 22, 'Rack': 23, 'Rolling_Bearing': 24, 'Rope_Thimbles': 25, 'Round_Nut': 26, 'Screw': 27, 'Spring': 28, 'Steel': 29,
                    'Stud': 30, 'Support': 31, 'TangentialKey': 32, 'Tee_Pipe': 33, 'TemplateAndPlate': 34, 'Thrust_Ring': 35, 'VacuumSealingElement': 36, 'Washer': 37, 'Wheel': 38, 'WireTensioner': 39}
text2int_cadnet10 = {'AngleIron': 0, 'FlatKey': 1, 'Grooved_Pin': 2, 'HookWrench': 3, 'Key': 4, 'Spring': 5,
                      'Steel': 6, 'TangentialKey': 7, 'Thrust_Ring': 8, 'Washer': 9}



def load_data(input_data, input_list):
    # Filter data list
    with open(input_list, "rb") as tf:
        data_list = pickle.load(tf)['train']

    data_paths = []
    data_classes = []
    for uid in data_list:
        try:
            path = os.path.join(input_data, str(math.floor(int(uid.split('.')[0])/10000)).zfill(4), uid)
            class_label = -1  # unconditional generation (abc/deepcad)
        except Exception:
            path = os.path.join(input_data, uid)  
            #class_label = text2int_cadnet10[uid.split('/')[0]]  # conditional generation (cadnet50)
            class_label = text2int[uid.split('/')[0]]  # conditional generation (furniture)
        data_paths.append(path)
        data_classes.append(class_label)
    return data_paths
def load_data_with_prefix(root_folder, prefix):
    data_files = []

    # Walk through the directory tree starting from the root folder
    for root, dirs, files in os.walk(root_folder):
        for filename in files:
            # Check if the file ends with the specified prefix
            if filename.endswith(prefix):
                file_path = os.path.join(root, filename)
                data_files.append(file_path)

    return data_files
def load_xyz(filename):
    category_arrays = []

    # 读取并解析文件
    with open(filename, 'r') as file:
        # 初始化一个空列表用于存储当前类别的点
        current_category_points = None
        current_category = None
        
        for line in file:
            # 去掉行首和行尾的空格
            line = line.strip()
            # 分割每一行的内容
            parts = line.split()
            
            # 获取坐标和类别
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            category = int(parts[3])
            
            # 如果当前类别不匹配，保存前一个类别的点并初始化新的类别
            if current_category != category:
                if current_category_points is not None:
                    # 保存前一个类别的点云
                    category_arrays.append(np.array(current_category_points))
                # 切换到新类别并初始化空点列表
                current_category = category
                current_category_points = [[x, y, z]]
            else:
                # 如果类别相同，继续添加点
                current_category_points.append([x, y, z])
        
        # 最后一类的保存
        if current_category_points is not None:
            category_arrays.append(np.array(current_category_points))
    return category_arrays
def compute_novel_unique(train_data, generated_data, n_bits=6):
    """
    计算Novel和Unique两个指标。
    
    :param train_data: 训练集，包含多个数据样本，每个样本有一个'surf_wcs'字段（该字段是一个列表）
    :param generated_data: 生成集，包含多个生成的数据样本，每个样本有一个'surf_wcs'字段（该字段是一个列表）
    :param n_bits: 量化的位数，默认是6位
    :return: Novel 和 Unique 的百分比
    """
    # Step 1: 对训练集数据进行哈希处理，存储每个样本的哈希值
    train_hashes = set()
    for data_path in train_data:
        with open(data_path, "rb") as tf:
            data = pickle.load(tf)
        surfs_wcs = data['surf_wcs']  # 这里的 surfs_wcs 是一个列表
        surf_hash_total = []
        for surf in surfs_wcs:
            np_bit = real2bit(surf, n_bits).reshape(-1, 3)  # bits
            data_hash = sha256(np_bit.tobytes()).hexdigest()
            surf_hash_total.append(data_hash)
        
        # 合并哈希值，得到唯一标识符
        data_hash = '_'.join(sorted(surf_hash_total))
        train_hashes.add(data_hash)

    # Step 2: 对生成集数据进行哈希处理，存储每个样本的哈希值
    generated_hashes = []
    for path in generated_data:
        surfs_wcs=load_xyz(path)
        #surfs_wcs = data['surf_wcs']  # 这里的 surfs_wcs 是一个列表
        surf_hash_total = []
        for surf in surfs_wcs:
            np_bit = real2bit(surf, n_bits).reshape(-1, 3)  # bits
            data_hash = sha256(np_bit.tobytes()).hexdigest()
            surf_hash_total.append(data_hash)
        
        # 合并哈希值，得到唯一标识符
        data_hash = '_'.join(sorted(surf_hash_total))
        generated_hashes.append(data_hash)

    # Step 3: 计算Novel和Unique
    novel_count = 0
    unique_count = 0
    unique_hash = set()
    
    for data_hash in generated_hashes:
        # Novel: 生成数据中不出现在训练集的数据
        if data_hash not in train_hashes:
            novel_count += 1
        # Unique: 生成数据中只出现一次的数据
        if data_hash not in unique_hash:
            unique_hash.add(data_hash)
            unique_count += 1
    
    # 计算Novel和Unique的百分比
    novel_percentage = (novel_count / len(generated_hashes)) * 100
    unique_percentage = (unique_count / len(generated_hashes)) * 100
    
    return novel_percentage, unique_percentage


input_list='data_process/cadnet40v2_split_6bit_filtered.pkl'
input_data='data_process/cadnet40v2_parsed'
gen_dir='/path/to/sample_cadiff/cadnet-2'
train_data=load_data(input_data, input_list, False)
generated_data=load_data_with_prefix(gen_dir,'xyzc')
print(len(generated_data))
# novel, unique = compute_novel_unique(train_data, generated_data)
# print(f"Novel: {novel:.2f}%")
# print(f"Unique: {unique:.2f}%")
