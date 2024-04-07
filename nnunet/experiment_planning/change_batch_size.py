from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np

if __name__ == '__main__':
    input_file = '/home/q/CPCANet/DATASET/nnUNet_preprocessed/Task002_Synapse/nnUNetPlansv2.1_plans_2D.pkl'
    output_file = '/home/q/CPCANet/DATASET/nnUNet_preprocessed/Task002_Synapse/nnUNetPlansv2.1_plans_2D.pkl'
    a = load_pickle(input_file)
    #a['plans_per_stage'][0]['batch_size'] = int(np.floor(6 / 9 * a['plans_per_stage'][0]['batch_size']))
    a['plans_per_stage'][0]['batch_size'] = 1
    save_pickle(a, output_file)