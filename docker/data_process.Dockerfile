ARG PYTORCH="1.6.0"
ARG CUDA="10.1"
ARG CUDNN="7"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

RUN pip install oss2 tqdm

ADD . /workspace
#ADD ./work_dirs/ossutil64 /bin/ossutil64
#ADD ./work_dirs/.ossutilconfig-ailab-car-sim /bin/.ossutilconfig-ailab-car-sim
ADD ./tools/autodrive_data_process.sh /bin/process.sh
RUN chmod +x /bin/process.sh
#RUN chmod +x /bin/ossutil64
