Apart from training/testing scripts, We provide lots of useful tools under the
 `tools/` directory.

## Log Analysis

`tools/analysis_tools/analyze_logs.py` plots loss/mAP curves given a training
 log file. Run `pip install seaborn` first to install the dependency.

 ```shell
python tools/analysis_tools/analyze_logs.py plot_curve [--keys ${KEYS}] [--title ${TITLE}] [--legend ${LEGEND}] [--backend ${BACKEND}] [--style ${STYLE}] [--out ${OUT_FILE}]
 ```

![loss curve image](../resources/loss_curve.png)

Examples:

- Plot the classification loss of some run.

    ```shell
    python tools/analysis_tools/analyze_logs.py plot_curve log.json --keys loss_cls --legend loss_cls
    ```

- Plot the classification and regression loss of some run, and save the figure to a pdf.

    ```shell
    python tools/analysis_tools/analyze_logs.py plot_curve log.json --keys loss_cls loss_bbox --out losses.pdf
    ```

- Compare the bbox mAP of two runs in the same figure.

    ```shell
    python tools/analysis_tools/analyze_logs.py plot_curve log1.json log2.json --keys bbox_mAP --legend run1 run2
    ```

- Compute the average training speed.

    ```shell
    python tools/analysis_tools/analyze_logs.py cal_train_time log.json [--include-outliers]
    ```

    The output is expected to be like the following.

    ```text
    -----Analyze train time of work_dirs/some_exp/20190611_192040.log.json-----
    slowest epoch 11, average time is 1.2024
    fastest epoch 1, average time is 1.1909
    time std over epochs is 0.0028
    average iter time: 1.1959 s/iter
    ```

## Result Analysis

`tools/analysis_tools/analyze_results.py` calculates single image mAP and saves or shows the topk images with the highest and lowest scores based on prediction results.

**Usage**

```shell
python tools/analysis_tools/analyze_results.py \
      ${CONFIG} \
      ${PREDICTION_PATH} \
      ${SHOW_DIR} \
      [--show] \
      [--wait-time ${WAIT_TIME}] \
      [--topk ${TOPK}] \
      [--show-score-thr ${SHOW_SCORE_THR}] \
      [--cfg-options ${CFG_OPTIONS}]
```

Description of all arguments:

- `config` : The path of a model config file.
- `prediction_path`:  Output result file in pickle format from `tools/test.py`
- `show_dir`: Directory where painted GT and detection images will be saved
- `--show`：Determines whether to show painted images, If not specified, it will be set to `False`
- `--wait-time`: The interval of show (s), 0 is block
- `--topk`: The number of saved images that have the highest and lowest `topk` scores after sorting. If not specified, it will be set to `20`.
- `--show-score-thr`:  Show score threshold. If not specified, it will be set to `0`.
- `--cfg-options`: If specified, the key-value pair optional cfg will be merged into config file

**Examples**:

Assume that you have got result file in pickle format from `tools/test.py`  in the path './result.pkl'.

1. Test Faster R-CNN and visualize the results, save images to the directory `results/`

```shell
python tools/analysis_tools/analyze_results.py \
       configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py \
       result.pkl \
       results \
       --show
```

2. Test Faster R-CNN and specified topk to 50, save images to the directory `results/`

```shell
python tools/analysis_tools/analyze_results.py \
       configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py \
       result.pkl \
       results \
       --topk 50
```

3. If you want to filter the low score prediction results, you can specify the `show-score-thr` parameter

```shell
python tools/analysis_tools/analyze_results.py \
       configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py \
       result.pkl \
       results \
       --show-score-thr 0.3
```

## Visualization

### Visualize Datasets

`tools/misc/browse_dataset.py` helps the user to browse a detection dataset (both
 images and bounding box annotations) visually, or save the image to a
  designated directory.

```shell
python tools/misc/browse_dataset.py ${CONFIG} [-h] [--skip-type ${SKIP_TYPE[SKIP_TYPE...]}] [--output-dir ${OUTPUT_DIR}] [--not-show] [--show-interval ${SHOW_INTERVAL}]
```

### Visualize Models

First, convert the model to ONNX as described
[here](#convert-mmdetection-model-to-onnx-experimental).
Note that currently only RetinaNet is supported, support for other models
 will be coming in later versions.
The converted model could be visualized by tools like [Netron](https://github.com/lutzroeder/netron).

### Visualize Predictions

If you need a lightweight GUI for visualizing the detection results, you can refer [DetVisGUI project](https://github.com/Chien-Hung/DetVisGUI/tree/mmdetection).

## Error Analysis

`tools/analysis_tools/coco_error_analysis.py` analyzes COCO results per category and by
 different criterion. It can also make a plot to provide useful information.

```shell
python tools/analysis_tools/coco_error_analysis.py ${RESULT} ${OUT_DIR} [-h] [--ann ${ANN}] [--types ${TYPES[TYPES...]}]
```

Example:

Assume that you have got [Mask R-CNN checkpoint file](http://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_1x_coco/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth) in the path 'checkpoint'. For other checkpoints, please refer to our [model zoo](./model_zoo.md). You can use the following command to get the results bbox and segmentation json file.

```shell
# out: results.bbox.json and results.segm.json
python tools/test.py \
       configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py \
       checkpoint/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth \
       --format-only \
       --options "jsonfile_prefix=./results"
```

1. Get COCO bbox error results per category , save analyze result images to the directory `results/`

```shell
python tools/analysis_tools/coco_error_analysis.py \
       results.bbox.json \
       results \
       --ann=data/coco/annotations/instances_val2017.json \
```

2. Get COCO segmentation error results per category , save analyze result images to the directory `results/`

```shell
python tools/analysis_tools/coco_error_analysis.py \
       results.segm.json \
       results \
       --ann=data/coco/annotations/instances_val2017.json \
       --types='segm'
```

## Model Serving

In order to serve an `MMDetection` model with [`TorchServe`](https://pytorch.org/serve/), you can follow the steps:

### 1. Convert model from MMDetection to TorchServe

```shell
python tools/deployment/mmdet2torchserve.py ${CONFIG_FILE} ${CHECKPOINT_FILE} \
--output_folder ${MODEL_STORE} \
--model-name ${MODEL_NAME}
```

### 2. Build `mmdet-serve` docker image

```shell
DOCKER_BUILDKIT=1 docker build -t mmdet-serve:latest docker/serve/
```

### 3. Launch `mmdet-serve`

Check the official docs for [running TorchServe with docker](https://github.com/pytorch/serve/blob/master/docker/README.md#running-torchserve-in-a-production-docker-environment).

Example:

```shell
docker run --rm \
--cpus 8 \
--gpus device=0 \
-p8080:8080 -p8081:8081 -p8082:8082 \
--mount type=bind,source=$MODEL_STORE,target=/home/model-server/model-store \
mmdet-serve:latest
```

***Note**: ${MODEL_STORE} needs to be an absolute path.

[Read the docs](https://github.com/pytorch/serve/blob/072f5d088cce9bb64b2a18af065886c9b01b317b/docs/rest_api.md) about the Inference (8080), Management (8081) and Metrics (8082) APis

### 4. Test deployment

```shell
curl -O curl -O https://raw.githubusercontent.com/pytorch/serve/master/docs/images/3dogs.jpg
curl http://127.0.0.1:8080/predictions/${MODEL_NAME} -T 3dogs.jpg
```

You should obtain a respose similar to:

```json
[
  {
    "dog": [
      402.9117736816406,
      124.19664001464844,
      571.7910766601562,
      292.6463623046875
    ],
    "score": 0.9561963081359863
  },
  {
    "dog": [
      293.90057373046875,
      196.2908477783203,
      417.4869079589844,
      286.2522277832031
    ],
    "score": 0.9179860353469849
  },
  {
    "dog": [
      202.178466796875,
      86.3709487915039,
      311.9863586425781,
      276.28411865234375
    ],
    "score": 0.8933767080307007
  }
]
```

## Model Complexity

`tools/analysis_tools/get_flops.py` is a script adapted from [flops-counter.pytorch](https://github.com/sovrasov/flops-counter.pytorch) to compute the FLOPs and params of a given model.

```shell
python tools/analysis_tools/get_flops.py ${CONFIG_FILE} [--shape ${INPUT_SHAPE}]
```

You will get the results like this.

```text
==============================
Input shape: (3, 1280, 800)
Flops: 239.32 GFLOPs
Params: 37.74 M
==============================
```

**Note**: This tool is still experimental and we do not guarantee that the
 number is absolutely correct. You may well use the result for simple
  comparisons, but double check it before you adopt it in technical reports or papers.

1. FLOPs are related to the input shape while parameters are not. The default
 input shape is (1, 3, 1280, 800).
2. Some operators are not counted into FLOPs like GN and custom operators. Refer to [`mmcv.cnn.get_model_complexity_info()`](https://github.com/open-mmlab/mmcv/blob/master/mmcv/cnn/utils/flops_counter.py) for details.
3. The FLOPs of two-stage detectors is dependent on the number of proposals.

## Model conversion

### MMDetection model to ONNX (experimental)

We provide a script to convert model to [ONNX](https://github.com/onnx/onnx) format. We also support comparing the output results between Pytorch and ONNX model for verification.

```shell
python tools/deployment/pytorch2onnx.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --output_file ${ONNX_FILE} [--shape ${INPUT_SHAPE} --verify]
```

**Note**: This tool is still experimental. Some customized operators are not supported for now. For a detailed description of the usage and the list of supported models, please refer to [pytorch2onnx](tutorials/pytorch2onnx.md).

### MMDetection 1.x model to MMDetection 2.x

`tools/model_converters/upgrade_model_version.py` upgrades a previous MMDetection checkpoint
 to the new version. Note that this script is not guaranteed to work as some
  breaking changes are introduced in the new version. It is recommended to
   directly use the new checkpoints.

```shell
python tools/model_converters/upgrade_model_version.py ${IN_FILE} ${OUT_FILE} [-h] [--num-classes NUM_CLASSES]
```

### RegNet model to MMDetection

`tools/model_converters/regnet2mmdet.py` convert keys in pycls pretrained RegNet models to
 MMDetection style.

```shell
python tools/model_converters/regnet2mmdet.py ${SRC} ${DST} [-h]
```

### Detectron ResNet to Pytorch

`tools/model_converters/detectron2pytorch.py` converts keys in the original detectron pretrained
 ResNet models to PyTorch style.

```shell
python tools/model_converters/detectron2pytorch.py ${SRC} ${DST} ${DEPTH} [-h]
```

### Prepare a model for publishing

`tools/model_converters/publish_model.py` helps users to prepare their model for publishing.

Before you upload a model to AWS, you may want to

1. convert model weights to CPU tensors
2. delete the optimizer states and
3. compute the hash of the checkpoint file and append the hash id to the
 filename.

```shell
python tools/model_converters/publish_model.py ${INPUT_FILENAME} ${OUTPUT_FILENAME}
```

E.g.,

```shell
python tools/model_converters/publish_model.py work_dirs/faster_rcnn/latest.pth faster_rcnn_r50_fpn_1x_20190801.pth
```

The final output filename will be `faster_rcnn_r50_fpn_1x_20190801-{hash id}.pth`.

## Dataset Conversion

`tools/data_converters/` contains tools to convert the Cityscapes dataset
 and Pascal VOC dataset to the COCO format.

```shell
python tools/dataset_converters/cityscapes.py ${CITYSCAPES_PATH} [-h] [--img-dir ${IMG_DIR}] [--gt-dir ${GT_DIR}] [-o ${OUT_DIR}] [--nproc ${NPROC}]
python tools/dataset_converters/pascal_voc.py ${DEVKIT_PATH} [-h] [-o ${OUT_DIR}]
```

## Miscellaneous

### Evaluating a metric

`tools/analysis_tools/eval_metric.py` evaluates certain metrics of a pkl result file
 according to a config file.

```shell
python tools/analysis_tools/eval_metric.py ${CONFIG} ${PKL_RESULTS} [-h] [--format-only] [--eval ${EVAL[EVAL ...]}]
                      [--cfg-options ${CFG_OPTIONS [CFG_OPTIONS ...]}]
                      [--eval-options ${EVAL_OPTIONS [EVAL_OPTIONS ...]}]
```

### Print the entire config

`tools/misc/print_config.py` prints the whole config verbatim, expanding all its
 imports.

```shell
python tools/misc/print_config.py ${CONFIG} [-h] [--options ${OPTIONS [OPTIONS...]}]
```

### Test the robustness of detectors

Please refer to [robustness_benchmarking.md](robustness_benchmarking.md).
