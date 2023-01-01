import os
import torch
import torch.nn as nn
import numpy as np
import logging
from torchnet.meter import ConfusionMeter
from dataloader2D import provider
import warnings
from progressbar import ProgressBar
import torchvision.models as models

warnings.simplefilter("ignore")
torch.backends.cudnn.benchmark = True


def calc_sensitivity(result, target):
    result = np.argmax(result, axis=1)
    result[result > 1] = 1
    target[target > 1] = 1
    numerator = np.sum(np.multiply(target, result))
    denominator = np.sum(target)
    if denominator == 0.0:
        return 1.0
    else:
        return numerator / denominator


def calc_specificity(result, target):
    result = np.argmax(result, axis=1)
    result[result > 1] = 1
    target[target > 1] = 1
    numerator = np.sum(np.multiply(target == 0, result == 0))
    denominator = np.sum(target == 0)
    if denominator == 0.0:
        return 1.0
    else:
        return numerator / denominator


def calc_precision(result, target):
    result = np.argmax(result, axis=1)
    result[result > 1] = 1
    target[target > 1] = 1
    numerator = np.sum(np.multiply(target, result))
    denominator = np.sum(np.multiply(target, result)) + np.sum(
        np.multiply(target == 0, result)
    )
    if denominator == 0.0:
        return 1.0
    else:
        return numerator / denominator


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def predict(net, config):

    confusion = ConfusionMeter(config["num_classes"])
    confusion.reset()

    logging.info("\nPredicting ...\n")

    net.eval()

    precisions, sensitivitys, specificitys = [], [], []

    data_loader = provider(config, mode="val")
    bar = ProgressBar()
    predict_dt = {}
    for batch in bar(data_loader):

        images, labels, ids = (
            batch["image"].float().cuda(non_blocking=True),
            batch["label"].long().cuda(non_blocking=True),
            batch["ID"],
        )

        with torch.no_grad():
            # try:
            output = net(images)

            soft_out = torch.softmax(output, dim=1)
            output = torch.argmax(soft_out, dim=1)
            output = output.view(-1).data.cpu()
            lbls = labels.data.cpu().numpy()
            labels = labels.view(-1).data.cpu()
            confusion.add(output, labels)

            out = soft_out.data.cpu().numpy()
            sensitivity = calc_sensitivity(out, lbls)
            precision = calc_precision(out, lbls)
            specificity = calc_specificity(out, lbls)
            sensitivitys.append(sensitivity)
            specificitys.append(specificity)
            precisions.append(precision)

            for i in range(len(ids)):
                predict_dt[ids[i]] = out[i]
            # except Exception as e:
            #     print(e)
            #     continue
    torch.cuda.empty_cache()
    print("Confusion Matrix -\n", confusion.conf)
    total_precision = np.array(precisions).mean()
    total_sensitivity = np.array(sensitivitys).mean()
    total_specificity = np.array(specificitys).mean()
    print(
        "Sensitivity - %.2f, Specificity - %.2f, Precision - %.2f"
        % (total_sensitivity, total_specificity, total_precision)
    )

    return predict_dt


if __name__ == "__main__":
    model_root = "/DDNstorage/users/krish/log/HeadNoHead/"
    exp_name = "exp1"
    model_name = "000_sensitivity.model"
    model_path = os.path.join(model_root, exp_name, model_name)

    ckpt = torch.load(model_path)
    config = ckpt["config"]

    config["val_batch_size"] = 64
    config["num_workers_val"] = 64
    config["val_list_path"] = "/DDNstorage/users/krish/head_clsfcn/val_files.npy"

    net = models.resnet34()
    net.conv1 = nn.Conv2d(
        1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
    )
    net.fc = nn.Linear(in_features=512, out_features=config["num_classes"])

    net.load_state_dict(ckpt["state_dict"])

    net = nn.DataParallel(net, [0])
    net.cuda()

    dt = predict(net, config)

    np.save(
        "/DDNstorage/users/krish/head_clsfcn/"
        + exp_name
        + "_"
        + model_name.split(".")[0]
        + ".npy",
        dt,
    )
