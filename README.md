# CA
The source code of paper "Cross-Domain Alignment for Zero-Shot Sketch-Based Image Retrieval"



============== Datasets: ==============

The datasets we used are provided by SAKE[1]. You can download the resized Sketchy Ext and TU-Berlin Ext dataset and train/test split files from the authors' GitHub Project. Then unzip the datasets files to the same directory ./dataset of this project.


[1] Liu, Qing, et al. "Semantic-aware knowledge preservation for zero-shot sketch-based image retrieval." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2019.


============== Training: ==============

cd src/

# train with Sketchy Ext dataset
python train_cse_resnet_sketchy_ext.py

# train with TU-Berlin Ext dataset
python train_cse_resnet_tuberlin_ext.py



============== Testing: ==============
# test with Sketchy Ext dataset
python test_cse_resnet_sketchy_zeroshot.py

# test with TU-Berlin Ext dataset
python test_cse_resnet_tuberlin_zeroshot.py
