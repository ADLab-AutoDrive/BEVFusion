import argparse
import mmcv
from mmcv import Config

from mmdet3d.datasets import build_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet3D visualize the results')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--result', help='results file in pickle format')
    parser.add_argument(
        '--show-dir', help='directory where visualize results will be saved')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    if args.result is not None and \
            not args.result.endswith(('.pkl', '.pickle')):
        raise ValueError('The results file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    cfg.data.test.test_mode = True

    # build the dataset
    dataset = build_dataset(cfg.data.test)
    results = mmcv.load(args.result)
    kwargs = {}

    eval_kwargs = cfg.get('evaluation', {}).copy()
    # hard-code way to remove EvalHook args
    for key in [
            'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
            'rule'
    ]:
        eval_kwargs.pop(key, None)
    eval_kwargs.update(dict(metric=args.eval, **kwargs))
    print(dataset.evaluate(results, **eval_kwargs))



if __name__ == '__main__':
    main()
