import os
import sys
import os.path as osp
# sys.path.insert(0, '/'.join(osp.realpath(__file__).split('/')[:-2]))
import logging
import numpy as np
import copy
import random
import json
import math
import itertools

import logging
# logger = logging.get_logger(__name__)

from utils.config import Config
from tools.videocomposer.inference_multi import inference_multi
from tools.videocomposer.inference_single import inference_single


def main():
    """
    Main function to spawn the train and test process.
    """
    cfg = Config(load=True)
    if hasattr(cfg, "TASK_TYPE") and cfg.TASK_TYPE == "MULTI_TASK":
        logging.info("TASK TYPE: %s " %  cfg.TASK_TYPE)
        inference_multi(cfg.cfg_dict)
    elif hasattr(cfg, "TASK_TYPE") and cfg.TASK_TYPE == "SINGLE_TASK":
        logging.info("TASK TYPE: %s " %  cfg.TASK_TYPE)
        inference_single(cfg.cfg_dict)
    else:
        logging.info('Not suport task %s' % (cfg.TASK_TYPE))

if __name__ == "__main__":
    main()
