import os
import torch
import torch.nn as nn
import numpy as np
import logging
from torchnet.meter import ConfusionMeter
from tensorboard_logger import configure, log_value
from dataloader2D import provider
import warnings
import torchvision.models as models

warnings.simplefilter("ignore")
torch.backends.cudnn.benchmark = True


logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s][%(asctime)s] %(message)s",
    datefmt="%I:%M:%S",
)


config = {}
config[
    "remarks"
] = "Training a head/no-head classification for PET on pulic datasets outside IRB"
config["save_dir"] = "/DDNstorage/users/krish/log/HeadNoHead/exp1_PT/"
config["train_list_path"] = "/DDNstorage/users/krish/head_clsfcn/train_files_PT.npy"
config["val_list_path"] = "/DDNstorage/users/krish/head_clsfcn/val_files_PT.npy"
config["train_batch_size"] = 256
config["val_batch_size"] = 512
config["num_gpus"] = [0, 1]
config["num_channels"] = 1
config["num_classes"] = 2
config["dropout"] = 0.25
config["learning_rate"] = 1e-4
config["weight_decay"] = 5e-5
config["num_workers_train"] = 64
config["num_workers_val"] = 100
config["zoom"] = 1
config["save_freq"] = 20
config["class_weights"] = [1.0, 4.0]


def dice(result, target):
    _, result = torch.max(result, dim=1)
    result, target = result.float(), target.float()
    numerator = 2 * (torch.dot(result, target))
    denominator = (torch.norm(target) ** 2) + (torch.norm(result) ** 2)
    return round((numerator / denominator), 2)


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


def dice_coeff(logits, true, eps=1e-7):
    logits = np.argmax(logits, axis=1)
    logits[logits > 1] = 1
    true[true > 1] = 1
    dice = (2 * np.sum(logits * true)) / (np.sum(logits) + np.sum(true) + eps)
    return dice


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def check_and_save_model(net, metric_name, metric, ckpt):

    save_model = False
    best_metric = ckpt["best_" + metric_name]
    save_dir = ckpt["config"]["save_dir"]
    epoch = ckpt["epoch"]
    save_freq = ckpt["config"]["save_freq"]

    if metric_name == "loss":
        if metric <= best_metric:
            save_model = True
            best_metric = metric
    else:
        if metric >= best_metric:
            save_model = True
            best_metric = metric

    if save_model:
        ckpt["best_" + metric_name] = best_metric

        state_dict = net.module.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()
        net_module = net.module.cpu()

        logging.info("Saving best " + metric_name + " model!")
        logging.info("best " + metric_name + " - " + str(best_metric))
        checkpoint = ckpt.copy()
        checkpoint["module"] = net_module
        checkpoint["state_dict"] = state_dict

        torch.save(
            checkpoint,
            os.path.join(
                save_dir,
                "%03d_" % int(int(epoch / save_freq) * save_freq)
                + metric_name
                + ".model",
            ),
        )

    return ckpt


def train(net, loss, optimizer, ckpt):

    config = ckpt["config"]
    epoch = ckpt["epoch"]
    save_dir = config["save_dir"]

    confusion = ConfusionMeter(config["num_classes"])
    confusion.reset()

    logging.info("\nTraining @ epoch: " + str(epoch))

    net.cuda()

    losses = []

    data_loader = provider(config, mode="train")
    total_train_batches = len(data_loader)
    # weight_tensor = torch.from_numpy(np.array(config["class_weights"])).cuda()

    for idx, batch in enumerate(data_loader):

        images, labels = (
            batch["image"].float().cuda(non_blocking=True),
            batch["label"].long().cuda(non_blocking=True),
        )

        with torch.set_grad_enabled(True):
            try:
                output = net(images)
                loss_output = loss(
                    output, labels,
                )
                loss_output = loss_output.mean()

                optimizer.zero_grad()
                loss_output.backward()

                optimizer.step()
            except Exception as e:
                print(e)
                continue

        loss_output = loss_output.item()
        losses.append(loss_output)

        if idx % 20 == 0:
            output = torch.argmax(torch.softmax(output, dim=1), dim=1)
            output = output.view(-1).data.cpu()
            labels = labels.view(-1).data.cpu()
            try:
                confusion.add(output, labels)
                logging.info("\n" + str(confusion.conf))
            except Exception as e:
                print(e)
            # if idx > 0:
            #     break

        logging.info(
            "[Tr - %d, batch %d/%d] loss: %.3f"
            % (epoch, idx, total_train_batches, loss_output)
        )

    torch.cuda.empty_cache()
    total_loss = np.array(losses).mean()

    log_value("Loss/Train loss", total_loss, epoch)

    checkpoint = ckpt.copy()
    state_dict = net.module.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].cpu()

    net_module = net.module
    net_module = net_module.cpu()

    checkpoint["state_dict"] = state_dict
    checkpoint["module"] = net_module

    torch.save(checkpoint, os.path.join(save_dir, "latest.ckpt"))
    net = net.cuda()
    return checkpoint


def validate(net, loss, ckpt):

    epoch = ckpt["epoch"]
    config = ckpt["config"]

    confusion = ConfusionMeter(config["num_classes"])
    confusion.reset()

    logging.info("\nValidating @ epoch: " + str(epoch))

    net.eval()

    losses, precisions, sensitivitys, specificitys = [], [], [], []

    data_loader = provider(config, mode="val")
    total_val_batches = len(data_loader)
    for idx, batch in enumerate(data_loader):

        images, labels = (
            batch["image"].float().cuda(non_blocking=True),
            batch["label"].long().cuda(non_blocking=True),
        )

        with torch.no_grad():
            try:
                output = net(images)
                loss_output = loss(output, labels).mean()
                losses.append(loss_output.item())

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
            except Exception as e:
                print(e)
                continue
            if idx % 20 == 0:
                logging.info("\n" + str(confusion.conf))
                # if idx > 0:
                #     break

        logging.info(
            "[Val - %d, batch %d/%d] loss: %.3f"
            % (epoch, idx, total_val_batches, loss_output)
        )
    torch.cuda.empty_cache()

    total_loss = np.array(losses).mean()
    total_precision = np.array(precisions).mean()
    total_sensitivity = np.array(sensitivitys).mean()
    total_specificity = np.array(specificitys).mean()

    log_value("Loss/ Val loss", total_loss, epoch)
    log_value("Metrics/Val sensitivity", total_sensitivity, epoch)
    log_value("Metrics/Val specificity", total_specificity, epoch)
    log_value("Metrics/Val precision", total_precision, epoch)

    ckpt = check_and_save_model(net, "loss", total_loss, ckpt)
    ckpt = check_and_save_model(net, "sensitivity", total_sensitivity, ckpt)
    ckpt = check_and_save_model(net, "specificity", total_specificity, ckpt)
    ckpt = check_and_save_model(net, "precision", total_precision, ckpt)

    net.cuda()

    return ckpt


if __name__ == "__main__":

    torch.manual_seed(0)

    save_dir = config["save_dir"]
    makedirs(save_dir)
    configure(save_dir, flush_secs=1)

    start_epoch = 0
    resume_training = False

    checkpoint = {}
    checkpoint["config"] = config
    checkpoint["best_loss"] = np.inf
    checkpoint["best_dice"] = 0.0
    checkpoint["best_specificity"] = 0.0
    checkpoint["best_sensitivity"] = 0.0
    checkpoint["best_precision"] = 0.0

    # net = seg_model.UNetResNet50(config["num_classes"], config["num_channels"])
    # net = seg_model.UNetResNext50(config["num_classes"], config["num_channels"])
    net = models.resnet34()
    net.conv1 = nn.Conv2d(
        1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
    )
    net.fc = nn.Linear(in_features=512, out_features=config["num_classes"])
    if resume_training:
        ckpt = torch.load(config["save_dir"] + "latest.ckpt")
        config1 = ckpt["config"]
        start_epoch = ckpt["epoch"] + 1
        net.load_state_dict(ckpt["state_dict"])
        checkpoint["best_loss"] = ckpt["best_loss"]
        checkpoint["best_specificity"] = ckpt["best_specificity"]
        checkpoint["best_sensitivity"] = ckpt["best_sensitivity"]
        checkpoint["best_precision"] = ckpt["best_precision"]

    net = nn.DataParallel(net, config["num_gpus"])
    loss = nn.CrossEntropyLoss(torch.FloatTensor(config["class_weights"]), reduce=False)

    net.cuda()
    loss.cuda()

    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, gamma=0.7)

    for epoch in range(start_epoch, 1000):
        checkpoint["epoch"] = epoch
        if epoch % config["save_freq"] == 0:
            checkpoint["best_loss"] = np.inf
            checkpoint["best_dice"] = 0.0
            checkpoint["best_specificity"] = 0.0
            checkpoint["best_sensitivity"] = 0.0
            checkpoint["best_precision"] = 0.0

        checkpoint = train(net, loss, optimizer, checkpoint)

        checkpoint = validate(net, loss, checkpoint)

        # scheduler.step()
