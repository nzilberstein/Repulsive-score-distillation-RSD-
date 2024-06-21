from matplotlib.pyplot import get
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from hydra.utils import call

from utils.checkpoints import ckpt_path_adm
from utils.distributed import get_logger


def build_model(cfg):
    logger = get_logger("model", cfg)
    model = call(cfg.model)
    map_location = {"cuda:0": f"cuda:{cfg.exp.gpu}"}
    model_ckpt = ckpt_path_adm(cfg.model.ckpt, cfg)
    logger.info(f"Loading model from {model_ckpt}..")
    model.load_state_dict(torch.load(model_ckpt, map_location=map_location))
    classifier = call(cfg.classifier)

    if getattr(cfg.classifier, "ckpt", None):
        classifier_ckpt = ckpt_path_adm(cfg.classifier.ckpt, cfg)
        logger.info(f"Loading classifier from {classifier_ckpt}..")
        classifier.load_state_dict(torch.load(classifier_ckpt))
    if classifier is not None:
        classifier.cuda(cfg.exp.gpu)
        # classifier = DDP(classifier, device_ids=[cfg.exp.gpu], output_device=[cfg.exp.gpu],)

    model.cuda(cfg.exp.gpu)
    # model = DDP(model, device_ids=[cfg.exp.gpu], output_device=[cfg.exp.gpu])
    return model, classifier
