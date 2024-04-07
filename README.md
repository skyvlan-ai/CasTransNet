# Channel prior convolutional attention for medical image segmentation

by Yong Wang, LuluZhang*, Yizhou Ding 

## Installation

```
git clone 
cd CasTransNet
conda env create -f environment.yml
source activate CasTransNet
pip install -e .
```

## Data-Preparation

CasTransNet is a 2D based network, and all data should be expressed in 2D form with ```.nii.gz``` format. You can download the original data from the link below. 

**Dataset I**
[ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/)

**Dataset II**
[Synapse](https://www.synapse.org/##!Synapse:syn3193805/wiki/217789)

The dataset should be finally organized as follows:

```
./DATASET/
  ├── nnUNet_raw/
      ├── nnUNet_raw_data/
          ├── Task01_ACDC/
              ├── imagesTr/
              ├── imagesTs/
              ├── labelsTr/
              ├── labelsTs/
              ├── dataset.json
              ├── evaulate.py

          ├── Task02_Synapse/
              ├── imagesTr/
              ├── imagesTs/
              ├── labelsTr/
              ├── labelsTs/
              ├── dataset.json
              ├── evaulate.py              
          ......
      ├── nnUNet_cropped_data/
  ├── nnUNet_trained_models/
  ├── nnUNet_preprocessed/
```

One thing you should be careful of is that folder imagesTr contains both training set and validation set, and correspondingly, the value of ```numTraining``` in dataset.json equals the case number in the imagesTr. The division of the training set and validation set will be done in the network configuration located at ```nnunet/network_configuration/config.py```.

## Data-Preprocessing

```
nnunet/experiment_planning/nnUNet_convert_decathlon_task.py -i (the input folder)
```

This step will convert the name of folder from Task01 to Task001, and make the name of each nifti files end with '_000x.nii.gz'.

```
nnunet/experiment_planning/nnUNet_plan_and_preprocess.py -t (id)
```

Where ```-t 1``` means the command will preprocess the data of the Task001_ACDC.
Before this step, you should set the environment variables to ensure the framework could know the path of ```nnUNet_raw```, ```nnUNet_preprocessed```, and ```nnUNet_trained_models```. The detailed construction can be found in [nnUNet](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/setting_up_paths.md).

## Training

If you want to train CasTransNet on ACDC.

```
nnunet/run/run_training.py - task 001
```

If you want to train CasTransNet on Synapse.

```
nnunet/run/run_training.py - task 002
```

## Testing

If you want to test CasTransNet on ACDC.

```
nnunet/inference/predict_simple.py -i (the ACDC input folder)
```

## Acknowledgements

Our code is origin from [nnUNet](https://github.com/MIC-DKFZ/nnUNet) and [TransCASCADE](https://github.com/SLDGroup/CASCADE). Thanks to these authors for their excellent work.
