from cfg.option import Options
from gv_tools.util.logger import Logger
import os
from modules.train import Trainer
import argparse


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'False'):
        return False


def get_args():
    args = argparse.ArgumentParser()
    args.add_argument('-cfg', '--config_file', type=str)
    args.add_argument('-s', '--save_path', type=str)
    args.add_argument('-social', '--social_only', default=False,
                      type=str2bool)
    args.add_argument('-m', '--model_type', default='M1',
                      type=str)
    args.add_argument('-p', '--pretrain_model', default=None,
                      type=str)
    args.add_argument('-ed', '--embedding_dim', default=64,
                      type=int)
    args.add_argument('-lr', '--lr', default=0.0001,
                      type=float)
    args.add_argument('-hd', '--hidden_dim', default=64,
                      type=int)
    args.add_argument('-dp', '--drop_out', default=0.0,
                      type=float)
    args.add_argument('-a', '--alpha', default=1.0,
                      type=float)
    args.add_argument('-b', '--beta', default=1.0,
                      type=float)
    return args.parse_args()

    # ------------------------------------------------------------------


if __name__ == "__main__":

    args = get_args()

    params = Options(args.config_file)
    logger = Logger()
    logger.attach_file_handler(args.save_path, "train")
    result_logger = Logger()
    result_logger.attach_file_handler(args.save_path, "results")
    t = Trainer(params, args, logger, result_logger)
    t.build_data_loader()
    t.build_model(args.pretrain_model)
    t.train()

