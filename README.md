# Cuneiform-Sign-Detection-Code

Author: Tobias Dencker - <tobias.dencker@gmail.com>

This is the code repository for the article submission on "Deep learning of cuneiform sign detection with weak supervision using transliteration alignment".

This repository contains code to execute the proposed iterative training procedure as well as code to evaluate and visualize results.
Moreover, we provide pre-trained models of the cuneiform sign detector for Neo-Assyrian script after iterative training on the [Cuneiform Sign Detection Dataset](https://compvis.github.io/cuneiform-sign-detection-dataset/).
Finally, we provide a web application for the analysis of tablet images with the help of a pre-trained cuneiform sign detector.

<img src="http://cunei.iwr.uni-heidelberg.de/cuneiformbrowser/functions/images_decent.jpg" alt="sign detections on tablet images: yellow box indicate TP and blue FP detections" width="700"/>
<!--- <img src="http://cunei.iwr.uni-heidelberg.de/cuneiformbrowser/functions/images_difficult.jpg" alt="Web interface detection" width="500"/> -->

## Repository description

- General structure:
    - `data`: tablet images, annotations, transliterations, metadata
    - `experiments`: training, testing, evaluation and visualization
    - `lib`: project library code
    - `results`: generated detections (placed, raw and aligned), network weights, logs
    - `scripts`: scripts to run the alignment and placement step of iterative training


### Use cases

- Pre-processing of training data
    - line detection
- Iterative training
    - generate sign annotations (aligned and placed detections)
    - sign detector training   
- Evaluation (on test set)
    - raw detections
    - placed detections
    - aligned detections
- Test & visualize
    - line segmentation and post-processing
    - line-level and sign-level alignments
    - TP/FP for raw, aligned and placed detections (full tablet and crop level)


### Pre-processing
As pre-processing of the training data line detections are obtained for all tablet images before iterative training.
- use jupyter notebooks (`experiments/line_segmentation/`) for train, eval of line segmentation network and to perform line detection on all tablet images of train set


### Training
*Iterative training* alternates between generating aligned and placed detections and training a new sign detector:
1. use command-line scripts (`scripts/generate/`) for running alignment and placement step of iterative training
2. use jupyter notebooks (`experiments/sign_detector/`) for sign detector training step of iterative training

To keep track of the sign detector and generated sign annotations of each iteration of iterative training (stored in `results/`),
we follow the convention to label the sign detector with a *model version* (e.g. v002)
which is also used to label the raw, aligned and placed detections based on this detector.
Besides providing a model version, a user also selects which subsets of the training data to use for the generation of new annotations.
In particular, *subsets of SAAo collections* (e.g. saa01, saa05, saa08) are selected, when running the scripts under `scripts/generate/`.
To enable the evaluation on the test set, it is necessary to include the collections (test, saa06).


### Evaluation
Use the [*test sign detector notebook*](./experiments/sign_detector/test_sign_detector.ipynb) in order to test the performance of the trained sign detector (mAP) on the test set or other subsets of the dataset.
In `experiments/alignment_evaluation/` you find further notebooks for evaluation and visualization of line-level and sign-level alignments and TP/FP for raw, aligned and placed detections (full tablet and crop level).


### Pre-trained models

We provide pre-trained models in the form of [PyTorch model files](https://pytorch.org/tutorials/beginner/saving_loading_models.html) for the line segmentation network as well as the sign detector.

| Model name     | Model type        | Train annotations  |
|----------------|-------------------|------------------------|
| [lineNet_basic_vpub.pth](http://cunei.iwr.uni-heidelberg.de/cuneiformbrowser/model_weights/lineNet_basic_vpub.pth) | line segmentation | 410 lines  |

For the sign detector, we provide the best weakly supervised model (fpn_net_vA) and the best semi-supervised model (fpn_net_vF).

| Model name     | Model type        | Weak supervision in training  | Annotations in training  |  mAP on test_full  |
|----------------|-------------------|-------------------|------------------------|------------------------|
| [fpn_net_vA.pth](http://cunei.iwr.uni-heidelberg.de/cuneiformbrowser/model_weights/fpn_net_vA.pth) | sign detector | saa01, saa05, saa08, saa10, saa13, saa16 | None  | 45.3  |
| [fpn_net_vF.pth](http://cunei.iwr.uni-heidelberg.de/cuneiformbrowser/model_weights/fpn_net_vF.pth) | sign detector | saa01, saa05, saa08, saa10, saa13, saa16 | train_full (4663 bboxes)  | 65.6  |




### Web application

We also provide a demo web application that enables a user to apply a trained cuneiform sign detector to a large collection of tablet images.
The code of the web front-end is available in the [webapp repo](https://github.com/compvis/cuneiform-sign-detection-webapp/).
The back-end code is part of this repository and is located in [lib/webapp/](./lib/webapp/).
Below you find a short animation of how the sign detector is used with this web interface.

<img src="http://cunei.iwr.uni-heidelberg.de/cuneiformbrowser/functions/demo_cuneiform_sign_detection.gif" alt="Web interface detection" width="700"/>


For demonstration purposes, we also host an instance of the web application: [Demo Web Application](http://cunei.iwr.uni-heidelberg.de/cuneiformbrowser/).
If you would like to test the web application, please contact us for user credentials to log in.
Please note that this web application is a prototype for demonstration purposes only and not a production system.
In case the website is not reachable, or other technical issues occur, please contact us.



### Cuneiform font

For visualization of the cuneiform characters, we recommend installing the [Unicode Cuneiform Fonts](https://www.hethport.uni-wuerzburg.de/cuneifont/) by Sylvie Vanseveren.


## Installation

#### Software
Install general dependencies:

- **OpenGM** with python wrapper - library for discrete graphical models. http://hciweb2.iwr.uni-heidelberg.de/opengm/  
This library is needed for the alignment step during training. Testing is not affected. An installation guide for Ubuntu 14.04 can be found [here](./install_opengm.md).

- Python 2.7.X

- Python packages:
    - torch 1.0
    - torchvision
    - scikit-image 0.14.0
    - pandas, scipy, sklearn, jupyter
    - pillow, tqdm, tensorboardX, nltk, Levensthein, editdistance, easydict


Clone this repository and place the [*cuneiform-sign-detection-dataset*](https://github.com/compvis/cuneiform-sign-detection-dataset) in the [./data sub-folder](./data/).

#### Hardware

Training and evaluation can be performed on a machine with a single GPU (we used a GeFore GTX 1080).
The demo web application can run on a web server without GPU support,
since detection inference with a lightweight MobileNetV2 backbone is fast even in CPU only mode
(less than 1s for an image with HD resolution, less than 10s for 4K resolution).

### References
This repository also includes external code. In particular, we want to mention:
> - kuangliu's *torchcv* and *pytorch-cifar* repositories from which we adapted the SSD and FPN detector code:
 https://github.com/kuangliu/pytorch-cifar and
 https://github.com/kuangliu/torchcv
> - Ross Girshick's *py-faster-rcnn* repository from which we adapted part of our evaluation routine:
 https://github.com/rbgirshick/py-faster-rcnn
> - Rico Sennrich's *Bleualign* repository from which we adapted part of the Bleualign implementation:
 https://github.com/rsennrich/Bleualign
