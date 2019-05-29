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

Blabla, video..... TODO! TODO!





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

- TODO!
***









### segmentation

- TODO!
***












### toyRegression

- Example usage:
```
$ sudo sh start_docker_image_toyProblems_depthCompletion.sh
$ cd --
$ python evaluating_bdl/toyRegression/Ensemble-Adam/train.py 
```
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
- - - _eval.py_: Creates one plot of the obtained predicitve distribution and the HMC "ground truth" predictive distribution, for a set value of M. Also creates histograms for the model parameters.
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
- - - TODO!.

- SGLD-64:
- - - TODO!.

- SGHMC-256:
- - - TODO!.

- SGHMC-64:
- - - TODO!.

- HMC:
- - - TODO!.
- - - _TODO.py_: TODO!
***
***
***
