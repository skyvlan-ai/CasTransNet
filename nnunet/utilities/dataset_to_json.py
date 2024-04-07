import os
import json
import SimpleITK as sitk

def is_valid_nii(file_path):
    try:
        sitk.ReadImage(file_path)
        return True
    except Exception as e:
        return False

def rename_files_in_folder(folder_path, prefix):
    # 遍历文件夹
    for root, dirs, files in os.walk(folder_path):
        for index, filename in enumerate(sorted(files), 1):
            if is_valid_nii(os.path.join(root, filename)):
                # 构造新的文件名，使用指定的前缀和索引
                new_filename = f"{prefix}{index:03d}.nii.gz"
                # 重命名文件
                os.rename(os.path.join(root, filename), os.path.join(root, new_filename))

def generate_dataset_json(data_folder, output_file):
    dataset = {
        "labels": {
        "0": "background",
        "1": "spleen",
        "2": "right kidney",
        "3": "left kidney",
        "4": "gallbladder",
        "5": "esophagus",
        "6": "liver",
        "7": "stomach",
        "8": "aorta",
        "9": "inferior vena cava",
        "10": "portal vein and splenic vein",
        "11": "pancreas",
        "12": "right adrenal gland",
        "13": "left adrenal gland"  
    },
        "licence": "hands off!",
        "modality": {
            "0": "CT"
        },  
        "name": "Synapse",
        "numTest": 0,
        "numTraining": 0,
        "description": "It seems that we use the whole data to train, but we will split the validation set from the training set",
        "reference": "see challenge website",
        "release": "0.0",
        "tensorImageSize": "4D",
        "test": [],
        "training": []
    }
    
    # 重命名训练数据文件
    rename_files_in_folder(os.path.join(data_folder, "imagesTr"), "img0")
    rename_files_in_folder(os.path.join(data_folder, "labelsTr"), "label0")
    rename_files_in_folder(os.path.join(data_folder, "imagesTs"), "img0")

    # 遍历训练数据文件夹
    for root, dirs, files in sorted(os.walk(os.path.join(data_folder, "imagesTr"))):
        for dir_name in sorted(files):
            if is_valid_nii(os.path.join(root, dir_name)):
                image_path = os.path.join(root, dir_name)
                # 查找相应的标签文件
                # label_filename = dir_name.replace("img", "label")
                label_path = os.path.join(data_folder, "labelsTr", dir_name)
                if os.path.isfile(label_path):
                    dataset["training"].append({
                        "image": f"./imagesTr/{dir_name}",
                        "label": f"./labelsTr/{dir_name}"
                    })
                    dataset["numTraining"] += 1
                
                

    # 遍历测试数据文件夹
    for root, dirs, files in sorted(os.walk(os.path.join(data_folder, "imagesTs"))):
        for dir_name in sorted(files):
            sample_path = os.path.join(root, dir_name)
            dataset["test"].append(f"./imagesTs/{dir_name}")  # 测试样本的相对路径
            dataset["numTest"] += 1  # 记录测试样本数量

    # 您可能需要手动添加标签和模态信息
    # dataset["labels"] = { "0": "label_0", "1": "label_1", ... }
    # dataset["modality"] = { "0": "MRI", "1": "CT", ... }

    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=4)


# 示例用法
data_folder = "/home/q/CPCANet/DATASET/nnUNet_raw/nnUNet_raw_data/Task02_Synapse"
output_file = "/home/q/CPCANet/DATASET/nnUNet_raw/nnUNet_raw_data/Task02_Synapse/dataset.json"
generate_dataset_json(data_folder, output_file)
print("ok")
