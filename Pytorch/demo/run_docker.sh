#!/bin/bash
sudo docker run \
--rm -it \
-p 8888:8888 \
-v $(pwd):/workdir \
-w /workdir \
--gpus all \
pytorch/pytorch \
"$@"