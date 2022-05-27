ARG PYTORCH="1.6.0"
ARG CUDA="10.1"
ARG CUDNN="7"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN apt-get update \
    && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 vim wget unzip sudo lsof libncurses5-dev libncursesw5-dev libssl-dev\
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://pitaya-data-for-annotation.oss-cn-hangzhou.aliyuncs.com/software/cmake-3.18.5.tar.gz \
    && tar zxvf cmake-3.18.5.tar.gz && rm -f cmake-3.18.5.tar.gz && cd cmake-3.18.5 \
    && ./bootstrap && make -j8 && make install
RUN rm -rf cmake-3.18.5

# Install MMCV, MMDetection and MMSegmentation
RUN pip install Cython
RUN pip install mmcv-full==1.3.13 -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.6.0/index.html
RUN pip install mmdet==2.17.0
RUN pip install mmsegmentation==0.18.0

# Install MMDetection3D
RUN conda clean --all
ADD . /workspace
WORKDIR /workspace
ENV FORCE_CUDA="1"
RUN pip install nuscenes-devkit einops
RUN pip install -r requirements/build.txt
#RUN python setup.py bdist_wheel
RUN pip install --no-cache-dir -e .
RUN pip uninstall mmdet3d --no-cache-dir -y
RUN rm -rf ./build
RUN find . -name "*.so" | xargs rm

# uninstall pycocotools installed by nuscenes-devkit and reinstall mmpycocotools
RUN pip uninstall pycocotools --no-cache-dir -y
RUN pip install mmpycocotools --no-cache-dir --force --no-deps

RUN pip install jupyterlab motmetrics timm
ADD ./docker/jupyter_notebook_config.json  /root/.jupyter/jupyter_notebook_config.json
ADD ./docker/notebook /bin/notebook
RUN chmod +x /bin/notebook

WORKDIR /
RUN rm -rf /workspace