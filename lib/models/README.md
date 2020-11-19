### Network architectures 

- `linenet.py` : a modified AlexNet used for line segmentation
- `mobilenetv2_mod03.py` : a modified MobileNetV2 used as backbone for the sign detector
- `mobilenetv2_fpn.py` : a FPN network wrapper for the backbone architecture
- `trained_model_loader.py` : contains functions to load sign detector and line segmentation models; 
describes how detectors are assembled from parts.