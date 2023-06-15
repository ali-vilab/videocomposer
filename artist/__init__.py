import os
import os.path as osp
import torch
import torch.distributed as dist
import oss2 as oss

__all__ = ['DOWNLOAD_TO_CACHE']


def DOWNLOAD_TO_CACHE(oss_key,
                      file_or_dirname=None,
                      cache_dir=osp.join('/'.join(osp.abspath(__file__).split('/')[:-2]), 'model_weights')):
    r"""Download OSS [file or folder] to the cache folder.
        Only the 0th process on each node will run the downloading.
        Barrier all processes until the downloading is completed.
    """
    # source and target paths
    base_path = osp.join(cache_dir, file_or_dirname or osp.basename(oss_key))
    
    return base_path
