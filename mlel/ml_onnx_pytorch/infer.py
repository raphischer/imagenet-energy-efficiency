import os
import warnings

import torch
import onnxruntime as rt
from torch import nn
import torch.utils.data
import torchvision

import mlel.ml_pytorch.pt_utils as utils
from mlel.ml_pytorch.pt_utils import model_name_mapping
from mlel.ml_pytorch.train import load_data


def _evaluate(model, criterion, data_loader, device, print_freq=100, log_suffix="", return_dict=False):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"Test: {log_suffix}"

    num_processed_samples = 0
    with torch.inference_mode():
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            output = torch.from_numpy(model.run(None, {'input': image.numpy()})[0])
            loss = criterion(output, target)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            num_processed_samples += batch_size
    # gather the stats from all processes

    num_processed_samples = utils.reduce_across_processes(num_processed_samples)
    if (
        hasattr(data_loader.dataset, "__len__")
        and len(data_loader.dataset) != num_processed_samples
        and torch.distributed.get_rank() == 0
    ):
        # See FIXME above
        warnings.warn(
            f"It looks like the dataset has {len(data_loader.dataset)} samples, but {num_processed_samples} "
            "samples were used for the validation, which might bias the results. "
            "Try adjusting the batch size and / or the world size. "
            "Setting the world size to 1 is always a safe bet."
        )

    metric_logger.synchronize_between_processes()

    loss = metric_logger.loss.global_avg
    acc1 = metric_logger.acc1.global_avg
    acc5 = metric_logger.acc5.global_avg

    if return_dict:
        return {"loss": loss, "accuracy": acc1/100, "top_5_accuracy": acc5/100}
    else:
        return loss, acc1, acc5


def init_inference(args, split):
    # TODO: Is it possible to seed onnx? 
    torch.manual_seed(args.seed)

    custom_trained = os.path.isdir(args.infer_model)

    # Set missing args, depending on model name
    args = utils.set_model_args(args)

    # Load data
    train_dir = os.path.join(args.data_path, "train")
    val_dir = os.path.join(args.data_path, "val")
    dataset_train, dataset_test, train_sampler, test_sampler = load_data(train_dir, val_dir, args)
    num_classes = len(dataset_train.classes)

    if split == "train":
        data_to_use = dataset_train
    else:
        data_to_use = dataset_test
    data_loader_test = torch.utils.data.DataLoader(data_to_use, batch_size=args.batch_size, sampler=test_sampler, num_workers=16, pin_memory=True, drop_last=True)

    # Create model
    if custom_trained:
        # Load weights from folder
        model = torchvision.models.__dict__[model_name_mapping[args.model]](pretrained=False, num_classes=num_classes)

        last_model =  sorted([f for f in os.listdir(args.infer_model) if f.startswith('checkpoint')])[-1]
        model.load_state_dict(torch.load(os.path.join(args.infer_model, last_model)))
    else:
        # Use pretrained weights
        model = torchvision.models.__dict__[model_name_mapping[args.model]](pretrained=True, num_classes=num_classes)

    # Convert to ONNX
    dummy_input = torch.randn(args.batch_size, 3, args.val_crop_size, args.val_crop_size).float()
    onnx_model_fname = os.path.join(args.output_dir, f"model.onnx")
    torch.onnx.export(model, dummy_input, onnx_model_fname, verbose=False, input_names=['input'], output_names=['output'])

    # Load into ONNX Runtime
    rt_session = rt.InferenceSession(onnx_model_fname)
    onnx_filesize = os.path.getsize(onnx_model_fname)

    model_info = {
        'params': sum(p.numel() for p in model.parameters()), # TODO: There appears to be no easy way to get nr_parameters of onnx model
        'fsize': onnx_filesize,
    }

    criterion = nn.CrossEntropyLoss(label_smoothing=0.0)

    eval_func = lambda: _evaluate(rt_session, criterion, data_loader_test, None, return_dict=True)
    return eval_func, model_info

def finalize_inference(results):
    return results
