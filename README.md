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
