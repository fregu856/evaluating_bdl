# evaluating_bdl

- My username on the server is "fregu482", i.e., my home folder is "/home/fregu482".

- $ sudo docker pull fregu856/evaluating_bdl:pytorch_pytorch_0.4_cuda9_cudnn7_evaluating_bdl
- Create start_docker_image_toyProblems_depthCompletion.sh containing:
```
#!/bin/bash

# DEFAULT VALUES
GPUIDS="0"
NAME="toyProblems_depthCompletion_GPU"

NV_GPU="$GPUIDS" nvidia-docker run -it --rm --shm-size 12G \
        -p 5700:5700\
        --name "$NAME""0" \
        -v /home/fregu482:/root/ \
        fregu856/evaluating_bdl:pytorch_pytorch_0.4_cuda9_cudnn7_evaluating_bdl bash
```
- (Inside the image, /root/ will now be mapped to /home/fregu482, i.e., $ cd -- takes you to the regular home folder.)

- (to create more containers, change "GPUIDS", "--name "$NAME""0"" and "-p 5700:5700")

- To start the image:
- - $ sudo sh start_docker_image_toyProblems_depthCompletion.sh
- To commit changes to the image:
- - Open a new terminal window.
- - $ sudo docker commit toyProblems_depthCompletion_GPU0 fregu856/evaluating_bdl:pytorch_pytorch_0.4_cuda9_cudnn7_evaluating_bdl
- To stop the image when it’s running:
- - $ sudo docker stop toyProblems_depthCompletion_GPU0
- To exit the image without killing running code:
- - Ctrl + P + Q
- To get back into a running image:
- - $ sudo docker attach toyProblems_depthCompletion_GPU0

```
$ sudo sh start_docker_image_toyProblems_depthCompletion.sh
$ cd --
$ python evaluating_bdl/toyClassification/Ensemble-Adam/train.py 
```





***
***
***
***
***
***
***
***
***






- My username on the server is "fregu482", i.e., my home folder is "/home/fregu482".

- $ sudo docker pull fregu856/evaluating_bdl:rainbowsecret_pytorch04_20180905_evaluating_bdl
- Create start_docker_image_segmentation.sh containing:
```      
#!/bin/bash

# DEFAULT VALUES
GPUIDS="0,1"
NAME="segmentation_GPU"

NV_GPU="$GPUIDS" nvidia-docker run -it --rm --shm-size 12G \
        -p 5900:5900 \
        --name "$NAME""01" \
        -v /home/fregu482:/home/ \
        fregu856/evaluating_bdl:rainbowsecret_pytorch04_20180905_evaluating_bdl bash
```
- (Inside the image, /home/ will now be mapped to /home/fregu482, i.e., $ cd home takes you to the regular home folder.)

- (to create more containers, change "GPUIDS", "--name "$NAME""01"" and "-p 5900:5900")

- To start the image:
- - $ sudo sh start_docker_image_segmentation.sh
- To commit changes to the image:
- - Open a new terminal window.
- - $ sudo docker commit segmentation_GPU01 fregu856/evaluating_bdl:rainbowsecret_pytorch04_20180905_evaluating_bdl
- To stop the image when it’s running:
- - $ sudo docker stop segmentation_GPU01
- To exit the image without killing running code:
- - Ctrl + P + Q
- To get back into a running image:
- - $ sudo docker attach segmentation_GPU01

```
$ sudo sh start_docker_image_segmentation.sh
$ cd home
$ /root/miniconda3/bin/python evaluating_bdl/segmentation/ensembling_train_syn.py
```










***
***
***
***
***
***
***
***
***

Blabla, video..... TODO! TODO!



## Acknowledgements

- The depthCompletion code is based on the implementation by [@fangchangma](https://github.com/fangchangma) found [here](https://github.com/fangchangma/self-supervised-depth-completion).
- The segmentation code is based on the implementation by [@PkuRainBow](https://github.com/PkuRainBow) found [here](https://github.com/PkuRainBow/OCNet.pytorch).





## Index
- [Usage](#usage)
- [Documentation](#documentation)
- - [depthCompletion](#depthcompletion)
- - [segmentation](#segmentation)
- - [toyRegression](#toyregression)
- - [toyClassification](#toyclassification)











***
***
***
## Usage:

- TODO!





segmentation:

- Download _resnet101-imagenet.pth_ from [here](http://sceneparsing.csail.mit.edu/model/pretrained_resnet/resnet101-imagenet.pth) and place it in _evaluating_bdl/segmentation_.

- Download Cityscapes by...and place it in...TODO!TODO!.
- Download Synscapes by...and place it in...TODO!TODO!
- Run _evaluating_bdl/segmentation/utils/preprocess_synscapes.py_ and...TODO!TODO!




***
***
***











***
***
***
## Documentation:

- [depthCompletion](#depthcompletion)
- [segmentation](#segmentation)
- [toyRegression](#toyregression)
- [toyClassification](#toyclassification)

### depthCompletion

- Example usage:
```
$ sudo sh start_docker_image_toyProblems_depthCompletion.sh
$ cd --
$ python evaluating_bdl/depthCompletion/ensembling_train_virtual.py
```

- _criterion.py_: Definitions of losses and metrics.
- _datasets.py_: Definitions of datasets, for KITTI depth completion (KITTI) and virtualKITTI.
- _model.py_: Definition of the CNN.
- _model_mcdropout.py_: Definition of the CNN, with inserted dropout layers.
- %%%%%


- _ensembling_train.py_: Code for training M _model.py_ models, on KITTI train.
- _ensembling_train_virtual.py_: As above, but on virtualKITTI train.

- _ensembling_eval.py_: Computes the loss and RMSE for a trained ensemble, on KITTI val. Also creates visualization images of the input data, ground truth, prediction and the estimated uncertainty. 
- _ensembling_eval_virtual.py_: As above, but on virtualKITTI val.

- _ensembling_eval_auce.py_: Computes the AUCE (mean +- std) for M = [1, 2, 4, 8, 16, 32] on KITTI val, based on a total of 33 trained ensemble members. Also creates calibration plots.
- _ensembling_eval_auce_virtual.py_: As above, but on virtualKITTI val.

- _ensembling_eval_ause.py_: Computes the AUSE (mean +- std) for M = [1, 2, 4, 8, 16, 32] on KITTI val, based on a total of 33 trained ensemble members. Also creates sparsification plots and sparsification error curves.
- _ensembling_eval_ause_virtual.py_: As above, but on virtualKITTI val.

- _ensembling_eval_seq.py_: Creates visualization videos (input data, ground truth, prediction and the estimated uncertainty) for a trained ensemble, on all sequences in KITTI val.
- _ensembling_eval_seq_virtual.py_: As above, but on all sequences in virtualKITTI val.
- %%%%%


- _mcdropout_train.py_: Code for training M _model_mcdropout.py_ models, on KITTI train.
- _mcdropout_train_virtual.py_: As above, but on virtualKITTI train.

- _mcdropout_eval.py_: Computes the loss and RMSE for a trained MC-dropout model with M forward passes, on KITTI val. Also creates visualization images of the input data, ground truth, prediction and the estimated uncertainty. 
- _mcdropout_eval_virtual.py_:  As above, but on virtualKITTI val.

- _mcdropout_eval_auce.py_: Computes the AUCE (mean +- std) for M = [1, 2, 4, 8, 16, 32] forward passes on KITTI val, based on a total of 16 trained MC-dropout models. Also creates calibration plots.
- _mcdropout_eval_auce_virtual.py_: As above, but on virtualKITTI val.

- _mcdropout_eval_ause.py_: Computes the AUSE (mean +- std) for M = [1, 2, 4, 8, 16, 32] forward passes on KITTI val, based on a total of 16 trained MC-dropout models. Also creates sparsification plots and sparsification error curves.
- _mcdropout_eval_ause_virtual.py_: As above, but on virtualKITTI val.

- _mcdropout_eval_seq.py_: Creates visualization videos (input data, ground truth, prediction and the estimated uncertainty) for a trained MC-dropout model with M forward passes, on all sequences in KITTI val.
- _mcdropout_eval_seq_virtual.py_: As above, but on all sequences in virtualKITTI val.
***
***
***















### segmentation

- Example usage:
```
$ sudo sh start_docker_image_segmentation.sh
$ cd home
$ /root/miniconda3/bin/python evaluating_bdl/segmentation/ensembling_train_syn.py
```

- models:
- - - model.py: TODO!
- - - model_mcdropout.py: TODO!
- - - aspp.py: TODO!
- - - resnet_block.py: TODO!



- utils:
- - - criterion.py: TODO!
- - - parallel.py: TODO!
- - - preprocess_synscapes.py: TODO!
- - - utils.py: TODO!



- _datasets.py_: TODO!
- %%%%%



- _ensembling_train.py_: (x)
- _ensembling_train_syn.py_: (x) 

- _ensembling_eval.py_: (x)
- _ensembling_eval_syn.py_: (x) 

- _ensembling_eval_ause_ece.py_: (x) 
- _ensembling_eval_ause_ece_syn.py_: (x)

- _ensembling_eval_seq.py_: FIXA!
- _ensembling_eval_seq_syn.py_: FIXA!  
- %%%%%



- _mcdropout_train.py_: (x) 
- _mcdropout_train_syn.py_: (x) 

- _mcdropout_eval.py_: (x)
- _mcdropout_eval_syn.py_: (x)

- _mcdropout_eval_ause_ece.py_: (x)
- _mcdropout_eval_ause_ece_syn.py_: (x) 

- _mcdropout_eval_seq.py_: FIXA!
- _mcdropout_eval_seq_syn.py_: FIXA!
***
***
***

















### toyRegression

- Example usage:
```
$ sudo sh start_docker_image_toyProblems_depthCompletion.sh
$ cd --
$ python evaluating_bdl/toyRegression/Ensemble-Adam/train.py 
```

- Ensemble-Adam:
- - Ensembling by minimizing the MLE objective using Adam and random initialization.
- - - _datasets.py_: Definition of the training dataset.
- - - _model.py_: Definition of the feed-forward neural network.
- - - _train.py_: Code for training M models.
- - - _eval.py_: Creates a plot of the obtained predicitve distribution and the HMC "ground truth" predictive distribution, for a set value of M. Also creates histograms for the model parameters.
- - - _eval_plots.py_: Creates plots of the obtained predictive distributions for different values of M.
- - - _eval_kl_div.py_: Computes the KL divergence between the obtained predictive distribution and the HMC "ground truth", for different values of M. 

- Ensemble-MAP-Adam:
- - - Ensembling by minimizing the MAP objective using Adam and random initialization.

- Ensemble-MAP-Adam-Fixed:
- - - Ensembling by minimizing the MAP objective using Adam and NO random initialization.

- Ensemble-MAP-SGD:
- - - Ensembling by minimizing the MAP objective using SGD and random initialization.

- Ensemble-MAP-SGDMOM:
- - - Ensembling by minimizing the MAP objective using SGDMOM and random initialization.

- MC-Dropout-MAP-02-Adam:
- - - MC-dropout by minimizing the MAP objective using Adam, p=0.2.

- MC-Dropout-MAP-02-SGD
- - - MC-dropout by minimizing the MAP objective using SGD, p=0.2.

- MC-Dropout-MAP-02-SGDMOM:
- - - MC-dropout by minimizing the MAP objective using SGDMOM, p=0.2.

- SGLD-256:
- - - Implementation of SGLD, trained for 256 times longer than each member of an ensemble.

- SGLD-64:
- - - Implementation of SGLD, trained for 64 times longer than each member of an ensemble..

- SGHMC-256:
- - - Implementation of SGHMC, trained for 256 times longer than each member of an ensemble.

- SGHMC-64:
- - - Implementation of SGHMC, trained for 64 times longer than each member of an ensemble.

- HMC:
- - - Implementation of HMC using [Pyro](http://pyro.ai/).

- Deterministic:
- - - Implementation of a fully deterministic model, i.e., direct regression.
***
***
***


















### toyClassification

- Example usage:
```
$ sudo sh start_docker_image_toyProblems_depthCompletion.sh
$ cd --
$ python evaluating_bdl/toyClassification/Ensemble-Adam/train.py 
```

- Ensemble-Adam:
- - Ensembling by minimizing the MLE objective using Adam and random initialization.
- - - _datasets.py_: Definition of the training dataset.
- - - _model.py_: Definition of the feed-forward neural network.
- - - _train.py_: Code for training M models.
- - - _eval.py_: Creates a plot of the obtained predicitve distribution and the HMC "ground truth" predictive distribution, for a set value of M. Also creates histograms for the model parameters.
- - - _eval_plots.py_: Creates plots of the obtained predictive distributions for different values of M.
- - - _eval_kl_div.py_: Computes the KL divergence between the obtained predictive distribution and the HMC "ground truth", for different values of M. 

- Ensemble-Adam-Fixed:
- - - Ensembling by minimizing the MLE objective using Adam and NO random initialization.

- Ensemble-MAP-Adam:
- - - Ensembling by minimizing the MAP objective using Adam and random initialization.

- Ensemble-MAP-SGD:
- - - Ensembling by minimizing the MAP objective using SGD and random initialization.

- Ensemble-MAP-SGDMOM:
- - - Ensembling by minimizing the MAP objective using SGDMOM and random initialization.

- MC-Dropout-MAP-01-Adam:
- - - MC-dropout by minimizing the MAP objective using Adam, p=0.1.

- MC-Dropout-MAP-02-SGD
- - - MC-dropout by minimizing the MAP objective using SGD, p=0.2.

- MC-Dropout-MAP-02-SGDMOM:
- - - MC-dropout by minimizing the MAP objective using SGDMOM, p=0.2.

- SGLD-256:
- - - Implementation of SGLD, trained for 256 times longer than each member of an ensemble.

- SGLD-64:
- - - Implementation of SGLD, trained for 64 times longer than each member of an ensemble..

- SGHMC-256:
- - - Implementation of SGHMC, trained for 256 times longer than each member of an ensemble.

- SGHMC-64:
- - - Implementation of SGHMC, trained for 64 times longer than each member of an ensemble.

- HMC:
- - - Implementation of HMC using [Pyro](http://pyro.ai/).
***
***
***
