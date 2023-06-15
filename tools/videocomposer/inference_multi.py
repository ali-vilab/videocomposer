import os
import os.path as osp
import sys
# append parent path to environment
sys.path.insert(0, '/'.join(osp.realpath(__file__).split('/')[:-4]))
import logging
import numpy as np
# import copy
from copy import deepcopy, copy
import random
import json
import math
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.cuda.amp as amp
import torchvision.transforms as T
import pynvml
import torchvision.transforms.functional as TF
from importlib import reload
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import clip_grad_norm_
import open_clip
from easydict import EasyDict
from collections import defaultdict
from functools import partial
from io import BytesIO
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from fairscale.nn.data_parallel import ShardedDataParallel
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from fairscale.optim.oss import OSS
from fairscale.optim.grad_scaler import ShardedGradScaler
from .datasets import VideoDataset
import artist.ops as ops
import artist.data as data

from artist import DOWNLOAD_TO_CACHE
from artist.models.clip import VisionTransformer
import artist.models as models
from .config import cfg
from .unet_sd import UNetSD_temporal
from einops import rearrange
from artist.optim import Adafactor, AnnealingLR
from .autoencoder import  AutoencoderKL, DiagonalGaussianDistribution
from tools.annotator.canny import CannyDetector
from tools.annotator.sketch import pidinet_bsd, sketch_simplification_gan
# from tools.annotator.histogram import Palette
from utils.config import Config
 

def beta_schedule(schedule, num_timesteps=1000, init_beta=None, last_beta=None):
    '''
    This code defines a function beta_schedule that generates a sequence of beta values based on the given input parameters. These beta values can be used in video diffusion processes. The function has the following parameters:
        schedule(str): Determines the type of beta schedule to be generated. It can be 'linear', 'linear_sd', 'quadratic', or 'cosine'.
        num_timesteps(int, optional): The number of timesteps for the generated beta schedule. Default is 1000.
        init_beta(float, optional): The initial beta value. If not provided, a default value is used based on the chosen schedule.
        last_beta(float, optional): The final beta value. If not provided, a default value is used based on the chosen schedule.
    The function returns a PyTorch tensor containing the generated beta values. The beta schedule is determined by the schedule parameter:
        1.Linear: Generates a linear sequence of beta values betweeninit_betaandlast_beta.
        2.Linear_sd: Generates a linear sequence of beta values between the square root of init_beta and the square root oflast_beta, and then squares the result.
        3.Quadratic: Similar to the 'linear_sd' schedule, but with different default values forinit_betaandlast_beta.
        4.Cosine: Generates a sequence of beta values based on a cosine function, ensuring the values are between 0 and 0.999.
    If an unsupported schedule is provided, a ValueError is raised with a message indicating the issue.
    '''
    if schedule == 'linear':
        scale = 1000.0 / num_timesteps
        init_beta = init_beta or scale * 0.0001
        last_beta = last_beta or scale * 0.02
        return torch.linspace(init_beta, last_beta, num_timesteps, dtype=torch.float64)
    elif schedule == 'linear_sd':
        return torch.linspace(init_beta ** 0.5, last_beta ** 0.5, num_timesteps, dtype=torch.float64) ** 2
    elif schedule == 'quadratic':
        init_beta = init_beta or 0.0015
        last_beta = last_beta or 0.0195
        return torch.linspace(init_beta ** 0.5, last_beta ** 0.5, num_timesteps, dtype=torch.float64) ** 2
    elif schedule == 'cosine':
        betas = []
        for step in range(num_timesteps):
            t1 = step / num_timesteps
            t2 = (step + 1) / num_timesteps
            fn = lambda u: math.cos((u + 0.008) / 1.008 * math.pi / 2) ** 2
            betas.append(min(1.0 - fn(t2) / fn(t1), 0.999))
        return torch.tensor(betas, dtype=torch.float64)
    else:
        raise ValueError(f'Unsupported schedule: {schedule}')


class FrozenOpenCLIPEmbedder(nn.Module):
    """
    Uses the OpenCLIP transformer encoder for text
    """
    LAYERS = [
        #"pooled",
        "last",
        "penultimate"
    ]
    def __init__(self, arch="ViT-H-14", pretrained="laion2b_s32b_b79k", device="cuda", max_length=77,
                 freeze=True, layer="last"):
        super().__init__()
        assert layer in self.LAYERS
        # 
        model, _, _ = open_clip.create_model_and_transforms(arch, device=torch.device('cpu'), pretrained=pretrained)
        # 
        del model.visual
        self.model = model

        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        if self.layer == "last":
            self.layer_idx = 0
        elif self.layer == "penultimate":
            self.layer_idx = 1
        else:
            raise NotImplementedError()

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        tokens = open_clip.tokenize(text)
        z = self.encode_with_transformer(tokens.to(self.device))
        return z

    def encode_with_transformer(self, text):
        x = self.model.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x)
        return x

    def text_transformer_forward(self, x: torch.Tensor, attn_mask = None):
        for i, r in enumerate(self.model.transformer.resblocks):
            if i == len(self.model.transformer.resblocks) - self.layer_idx:
                break
            if self.model.transformer.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(r, x, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x

    def encode(self, text):
        return self(text)

def load_stable_diffusion_pretrained(state_dict, temporal_attention):
    import collections
    sd_new = collections.OrderedDict()
    keys = list(state_dict.keys())

    # "input_blocks.3.op.weight", "input_blocks.3.op.bias", "input_blocks.6.op.weight", "input_blocks.6.op.bias", "input_blocks.9.op.weight", "input_blocks.9.op.bias". 
    # "input_blocks.3.0.op.weight", "input_blocks.3.0.op.bias", "input_blocks.6.0.op.weight", "input_blocks.6.0.op.bias", "input_blocks.9.0.op.weight", "input_blocks.9.0.op.bias".
    for k in keys:
        if k.find('diffusion_model') >= 0:
            k_new = k.split('diffusion_model.')[-1]
            if k_new in ["input_blocks.3.0.op.weight", "input_blocks.3.0.op.bias", "input_blocks.6.0.op.weight", "input_blocks.6.0.op.bias", "input_blocks.9.0.op.weight", "input_blocks.9.0.op.bias"]:
                k_new = k_new.replace('0.op','op')
            if temporal_attention:
                if k_new.find('middle_block.2') >= 0:
                    k_new = k_new.replace('middle_block.2','middle_block.3')
                if k_new.find('output_blocks.5.2') >= 0:
                    k_new = k_new.replace('output_blocks.5.2','output_blocks.5.3')
                if k_new.find('output_blocks.8.2') >= 0:
                    k_new = k_new.replace('output_blocks.8.2','output_blocks.8.3')
            sd_new[k_new] = state_dict[k]

    return sd_new

def random_resize(img, size):
    img = [TF.resize(u, size, interpolation=random.choice([
        InterpolationMode.BILINEAR,
        InterpolationMode.BICUBIC,
        InterpolationMode.LANCZOS])) for u in img]
    return img

class CenterCrop(object):

    def __init__(self, size):
        self.size = size
    
    def __call__(self, img):
        # fast resize
        while min(img.size) >= 2 * self.size:
            img = img.resize((img.width // 2, img.height // 2), resample=Image.BOX)
        scale = self.size / min(img.size)
        img = img.resize((round(scale * img.width), round(scale * img.height)), resample=Image.BICUBIC)

        # center crop
        x1 = (img.width - self.size) // 2
        y1 = (img.height - self.size) // 2
        img = img.crop((x1, y1, x1 + self.size, y1 + self.size))
        return img

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.1):
        self.std = std
        self.mean = mean
    def __call__(self, img):
        assert isinstance(img, torch.Tensor)
        dtype = img.dtype
        if not img.is_floating_point():
            img = img.to(torch.float32)
        out = img + self.std * torch.randn_like(img) + self.mean        
        if out.dtype != dtype:
            out = out.to(dtype)
        return out
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def make_masked_images(imgs, masks):
    masked_imgs = []
    for i, mask in enumerate(masks):        
        # concatenation
        masked_imgs.append(torch.cat([imgs[i] * (1 - mask), (1 - mask)], dim=1))
    return torch.stack(masked_imgs, dim=0)

@torch.no_grad()
def get_first_stage_encoding(encoder_posterior):
    scale_factor = 0.18215                                                                     
    if isinstance(encoder_posterior, DiagonalGaussianDistribution):
        z = encoder_posterior.sample()
    elif isinstance(encoder_posterior, torch.Tensor):
        z = encoder_posterior
    else:
        raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
    return scale_factor * z


class FrozenOpenCLIPVisualEmbedder(nn.Module):
    """
    Uses the OpenCLIP transformer encoder for text
    """
    LAYERS = [
        #"pooled",
        "last",
        "penultimate"
    ]
    def __init__(self, arch="ViT-H-14", pretrained="laion2b_s32b_b79k", device="cuda", max_length=77,
                 freeze=True, layer="last", input_shape=(224, 224, 3)):
        super().__init__()
        assert layer in self.LAYERS
        # version = 'cache/open_clip_pytorch_model.bin'
        model, _, preprocess = open_clip.create_model_and_transforms(arch, device=torch.device('cpu'), pretrained=pretrained) # '/mnt/workspace/videocomposer/VideoComposer_diffusion/cache/open_clip_pytorch_model.bin'
        # model, _, _ = open_clip.create_model_and_transforms(arch, device=device, pretrained=version)
        del model.transformer 
        self.model = model
        data_white=np.ones(input_shape, dtype=np.uint8) * 255
        self.black_image = preprocess(T.ToPILImage()(data_white)).unsqueeze(0)

        self.device = device
        self.max_length = max_length # 77
        if freeze:
            self.freeze()
        self.layer = layer # 'penultimate'
        if self.layer == "last":
            self.layer_idx = 0
        elif self.layer == "penultimate":
            self.layer_idx = 1
        else:
            raise NotImplementedError()

    def freeze(self): 
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, image):
        # tokens = open_clip.tokenize(text)
        z = self.model.encode_image(image.to(self.device))
        return z

    def encode_with_transformer(self, text):
        x = self.model.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x)
        return x

    def text_transformer_forward(self, x: torch.Tensor, attn_mask = None):
        for i, r in enumerate(self.model.transformer.resblocks):
            if i == len(self.model.transformer.resblocks) - self.layer_idx:
                break
            if self.model.transformer.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(r, x, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x

    def encode(self, text):
        return self(text)


def find_free_port():
    """ https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number """
    import socket
    from contextlib import closing
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


def inference_multi(cfg_update, **kwargs):
    cfg.update(**kwargs)
    
    # Copy update input parameter to current task
    for k, v in cfg_update.items():
        cfg[k] = v

    if not 'MASTER_ADDR' in os.environ:
        os.environ['MASTER_ADDR']='localhost'
        os.environ['MASTER_PORT']= find_free_port()
    cfg.pmi_rank = int(os.getenv('RANK', 0)) # 0
    cfg.pmi_world_size = int(os.getenv('WORLD_SIZE', 1))
    setup_seed(cfg.seed)

    if cfg.debug:
        cfg.gpus_per_machine = 1
        cfg.world_size = 1
    else:
        cfg.gpus_per_machine = torch.cuda.device_count()
        cfg.world_size = cfg.pmi_world_size * cfg.gpus_per_machine
    
    if cfg.world_size == 1:
        worker(0, cfg)
    else:
        mp.spawn(worker, nprocs=cfg.gpus_per_machine, args=(cfg, ))
    return cfg

def worker(gpu, cfg):
    cfg.gpu = gpu
    cfg.rank = cfg.pmi_rank * cfg.gpus_per_machine + gpu

    # init distributed processes
    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True
    if not cfg.debug:
        dist.init_process_group(backend='nccl', world_size=cfg.world_size, rank=cfg.rank)

    # logging
    input_video_name = os.path.basename(cfg.input_video).split('.')[0]
    log_dir = ops.generalized_all_gather(cfg.log_dir)[0]
    exp_name = os.path.basename(cfg.cfg_file).split('.')[0] + f"-{input_video_name}" + '-S%05d' % (cfg.seed)
    log_dir = os.path.join(log_dir, exp_name)
    os.makedirs(log_dir, exist_ok=True)
    cfg.log_dir = log_dir
    if cfg.rank == 0:
        name = osp.basename(cfg.log_dir)
        cfg.log_file = osp.join(cfg.log_dir, '{}_rank{}.log'.format(name, cfg.rank))
        reload(logging)
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s] %(levelname)s: %(message)s',
            handlers=[
                logging.FileHandler(filename=cfg.log_file),
                logging.StreamHandler(stream=sys.stdout)])
        logging.info(cfg)

    # rank-wise params
    l1 = len(cfg.frame_lens)
    l2 = len(cfg.feature_framerates)
    cfg.max_frames = cfg.frame_lens[cfg.rank % (l1*l2)// l2]
    cfg.batch_size = cfg.batch_sizes[str(cfg.max_frames)]
    
    # [Transformer] Transformers for different inputs
    infer_trans = data.Compose([
        data.CenterCropV2(size=cfg.resolution),
        data.ToTensor(),
        data.Normalize(mean=cfg.mean, std=cfg.std)])

    misc_transforms = data.Compose([
        T.Lambda(partial(random_resize, size=cfg.misc_size)),
        data.CenterCropV2(cfg.misc_size),
        data.ToTensor()])

    mv_transforms = data.Compose([
        T.Resize(size=cfg.resolution),
        T.CenterCrop(cfg.resolution)])

    dataset = VideoDataset(
        cfg=cfg,
        max_words=cfg.max_words,
        feature_framerate=cfg.feature_framerate,
        max_frames=cfg.max_frames,
        image_resolution=cfg.resolution,
        transforms=infer_trans,
        mv_transforms=mv_transforms,
        misc_transforms=misc_transforms,
        vit_transforms=T.Compose([
            CenterCrop(cfg.vit_image_size),
            T.ToTensor(),
            T.Normalize(mean=cfg.vit_mean, std=cfg.vit_std)]),
        vit_image_size= cfg.vit_image_size,
        misc_size=cfg.misc_size)

    dataloader = DataLoader(
        dataset=dataset,
        num_workers=0,
        pin_memory=True)

    clip_encoder = FrozenOpenCLIPEmbedder(layer='penultimate',pretrained = DOWNLOAD_TO_CACHE(cfg.clip_checkpoint))
    clip_encoder.model.to(gpu)
    zero_y = clip_encoder("").detach() # [1, 77, 1024]
    
    clip_encoder_visual = FrozenOpenCLIPVisualEmbedder(layer='penultimate',pretrained = DOWNLOAD_TO_CACHE(cfg.clip_checkpoint))
    clip_encoder_visual.model.to(gpu)
    black_image_feature = clip_encoder_visual(clip_encoder_visual.black_image).unsqueeze(1) # [1, 1, 1024]
    black_image_feature = torch.zeros_like(black_image_feature) # for old
    
    # [Contions] Generators for various conditions
    if 'depthmap' in cfg.video_compositions:
        midas = models.midas_v3(pretrained=True).eval().requires_grad_(False).to(
            memory_format=torch.channels_last).half().to(gpu)
    if 'canny' in cfg.video_compositions:
        canny_detector = CannyDetector()
    if 'sketch' in cfg.video_compositions:
        pidinet = pidinet_bsd(pretrained=True, vanilla_cnn=True).eval().requires_grad_(False).to(gpu)
        cleaner = sketch_simplification_gan(pretrained=True).eval().requires_grad_(False).to(gpu)
        pidi_mean = torch.tensor(cfg.sketch_mean).view(1, -1, 1, 1).to(gpu)
        pidi_std = torch.tensor(cfg.sketch_std).view(1, -1, 1, 1).to(gpu)
    # Placeholder for color inference
    palette = None

    # [model] auotoencoder
    ddconfig = {'double_z': True, 'z_channels': 4, \
                'resolution': 256, 'in_channels': 3, \
                'out_ch': 3, 'ch': 128, 'ch_mult': [1, 2, 4, 4], \
                'num_res_blocks': 2, 'attn_resolutions': [], 'dropout': 0.0}
    autoencoder = AutoencoderKL(ddconfig, 4, ckpt_path=DOWNLOAD_TO_CACHE(cfg.sd_checkpoint))
    autoencoder.eval()
    for param in autoencoder.parameters():
        param.requires_grad = False
    autoencoder.cuda()
    
    if hasattr(cfg, "network_name") and cfg.network_name == "UNetSD_temporal":
        model = UNetSD_temporal(
            cfg=cfg,
            in_dim=cfg.unet_in_dim,
            concat_dim= cfg.unet_concat_dim,
            dim=cfg.unet_dim,
            y_dim=cfg.unet_y_dim,
            context_dim=cfg.unet_context_dim,
            out_dim=cfg.unet_out_dim,
            dim_mult=cfg.unet_dim_mult,
            num_heads=cfg.unet_num_heads,
            head_dim=cfg.unet_head_dim,
            num_res_blocks=cfg.unet_res_blocks,
            attn_scales=cfg.unet_attn_scales,
            dropout=cfg.unet_dropout,
            temporal_attention = cfg.temporal_attention,
            temporal_attn_times = cfg.temporal_attn_times,
            use_checkpoint=cfg.use_checkpoint,
            use_fps_condition=cfg.use_fps_condition,
            use_sim_mask=cfg.use_sim_mask,
            video_compositions=cfg.video_compositions,
            misc_dropout=cfg.misc_dropout,
            p_all_zero=cfg.p_all_zero,
            p_all_keep=cfg.p_all_zero,
            zero_y = zero_y,
            black_image_feature = black_image_feature,
            ).to(gpu)
    else:
        logging.info("Other model type not implement, exist")
        raise NotImplementedError(f"The model {cfg.network_name} not implement")
        return 

    # Load checkpoint
    resume_step = 1
    if cfg.resume and cfg.resume_checkpoint:
        if hasattr(cfg, "text_to_video_pretrain") and cfg.text_to_video_pretrain:
            ss = torch.load(DOWNLOAD_TO_CACHE(cfg.resume_checkpoint))
            ss = {key:p for key,p in ss.items() if 'input_blocks.0.0' not in key}
            model.load_state_dict(ss,strict=False)
        else:
            model.load_state_dict(torch.load(DOWNLOAD_TO_CACHE(cfg.resume_checkpoint), map_location='cpu'),strict=False)
        if cfg.resume_step:
            resume_step = cfg.resume_step

        logging.info(f'Successfully load step {resume_step} model from {cfg.resume_checkpoint}')
        torch.cuda.empty_cache()
    else:
        logging.error(f'The checkpoint file {cfg.resume_checkpoint} is wrong')
        raise ValueError(f'The checkpoint file {cfg.resume_checkpoint} is wrong ')
        return
    
    # mark model size
    if cfg.rank == 0:
        logging.info(f'Created a model with {int(sum(p.numel() for p in model.parameters()) / (1024 ** 2))}M parameters')

    # diffusion
    betas = beta_schedule('linear_sd', cfg.num_timesteps, init_beta=0.00085, last_beta=0.0120)
    diffusion = ops.GaussianDiffusion(
        betas=betas,
        mean_type=cfg.mean_type,
        var_type=cfg.var_type,
        loss_type=cfg.loss_type,
        rescale_timesteps=False)

    # global variables
    viz_num = cfg.batch_size
    for step, batch in enumerate(dataloader):
        model.eval() 

        caps = batch[1]; del batch[1]
        batch = ops.to_device(batch, gpu, non_blocking=True)
        if cfg.max_frames == 1 and cfg.use_image_dataset:
            ref_imgs, video_data, misc_data, mask, mv_data = batch
            fps =  torch.tensor([cfg.feature_framerate]*cfg.batch_size,dtype=torch.long, device=gpu)
        else:
            ref_imgs, video_data, misc_data, fps, mask, mv_data = batch

        ### save for visualization
        misc_backups = copy(misc_data)
        misc_backups = rearrange(misc_backups, 'b f c h w -> b c f h w')
        mv_data_video = []
        if 'motion' in cfg.video_compositions:
            mv_data_video = rearrange(mv_data, 'b f c h w -> b c f h w')

        ### mask images
        masked_video = []
        if 'mask' in cfg.video_compositions:
            masked_video = make_masked_images(misc_data.sub(0.5).div_(0.5), mask)
            masked_video = rearrange(masked_video, 'b f c h w -> b c f h w')
        
        image_local = []
        if 'local_image' in cfg.video_compositions:
            frames_num = misc_data.shape[1]
            bs_vd_local = misc_data.shape[0]
            image_local = misc_data[:,:1].clone().repeat(1,frames_num,1,1,1)
            image_local = rearrange(image_local, 'b f c h w -> b c f h w', b = bs_vd_local)
        
        ### encode the video_data
        bs_vd = video_data.shape[0]
        video_data_origin = video_data.clone() 
        video_data = rearrange(video_data, 'b f c h w -> (b f) c h w')
        misc_data = rearrange(misc_data, 'b f c h w -> (b f) c h w')
        # video_data_origin = video_data.clone() 

        video_data_list = torch.chunk(video_data, video_data.shape[0]//cfg.chunk_size,dim=0)
        misc_data_list = torch.chunk(misc_data, misc_data.shape[0]//cfg.chunk_size,dim=0)

        with torch.no_grad():
            decode_data = []
            for vd_data in video_data_list:
                encoder_posterior = autoencoder.encode(vd_data)
                tmp = get_first_stage_encoding(encoder_posterior).detach()
                decode_data.append(tmp)
            video_data = torch.cat(decode_data,dim=0)
            video_data = rearrange(video_data, '(b f) c h w -> b c f h w', b = bs_vd)

            depth_data = []
            if 'depthmap' in cfg.video_compositions:
                for misc_imgs in misc_data_list:
                    depth = midas(misc_imgs.sub(0.5).div_(0.5).to(memory_format=torch.channels_last).half())
                    depth = (depth / cfg.depth_std).clamp_(0, cfg.depth_clamp)
                    depth_data.append(depth)
                depth_data = torch.cat(depth_data, dim = 0)
                depth_data = rearrange(depth_data, '(b f) c h w -> b c f h w', b = bs_vd)
            
            canny_data = []
            if 'canny' in cfg.video_compositions:
                for misc_imgs in misc_data_list:
                    # print(misc_imgs.shape)
                    misc_imgs = rearrange(misc_imgs.clone(), 'k c h w -> k h w c') # 'k' means 'chunk'.
                    canny_condition = torch.stack([canny_detector(misc_img) for misc_img in misc_imgs])
                    canny_condition = rearrange(canny_condition, 'k h w c-> k c h w')
                    canny_data.append(canny_condition)
                canny_data = torch.cat(canny_data, dim = 0)
                canny_data = rearrange(canny_data, '(b f) c h w -> b c f h w', b = bs_vd)
            
            sketch_data = []
            if 'sketch' in cfg.video_compositions:
                for misc_imgs in misc_data_list:
                    sketch = pidinet(misc_imgs.sub(pidi_mean).div_(pidi_std))
                    sketch = 1.0 - cleaner(1.0 - sketch)
                    sketch_data.append(sketch)
                sketch_data = torch.cat(sketch_data, dim = 0)
                sketch_data = rearrange(sketch_data, '(b f) c h w -> b c f h w', b = bs_vd)

            single_sketch_data = []
            if 'single_sketch' in cfg.video_compositions:
                single_sketch_data = sketch_data.clone()[:,:,:1].repeat(1,1,frames_num,1,1)

        # preprocess for input text descripts
        y = clip_encoder(caps).detach()  # [1, 77, 1024]
        y0 = y.clone()
        
        y_visual = []
        if 'image' in cfg.video_compositions:
            with torch.no_grad():
                ref_imgs = ref_imgs.squeeze(1)
                y_visual = clip_encoder_visual(ref_imgs).unsqueeze(1) # [1, 1, 1024]
                y_visual0 = y_visual.clone()

        with torch.no_grad():
            # Log memory
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            logging.info(f'GPU Memory used {meminfo.used / (1024 ** 3):.2f} GB')
            # Sample images (DDIM)
            with amp.autocast(enabled=cfg.use_fp16):
                if cfg.share_noise:
                    b, c, f, h, w = video_data.shape
                    noise = torch.randn((viz_num, c, h, w), device=gpu)
                    noise = noise.repeat_interleave(repeats=f, dim=0) 
                    noise = rearrange(noise, '(b f) c h w->b c f h w', b = viz_num) 
                    noise = noise.contiguous()
                else:
                    noise=torch.randn_like(video_data[:viz_num])

                full_model_kwargs=[
                    {'y': y0[:viz_num],
                    "local_image": None if len(image_local) == 0 else image_local[:viz_num],
                    'image': None if len(y_visual) == 0 else y_visual0[:viz_num],
                    'depth': None if len(depth_data) == 0 else depth_data[:viz_num],
                    'canny': None if len(canny_data) == 0 else canny_data[:viz_num],
                    'sketch': None if len(sketch_data) == 0 else sketch_data[:viz_num],
                    'masked': None if len(masked_video) == 0 else masked_video[:viz_num],
                    'motion': None if len(mv_data_video) == 0 else mv_data_video[:viz_num],
                    'single_sketch': None if len(single_sketch_data) == 0 else single_sketch_data[:viz_num],
                    'fps': fps[:viz_num]}, 
                    {'y': zero_y.repeat(viz_num,1,1) if not cfg.use_fps_condition else torch.zeros_like(y0)[:viz_num],
                    "local_image": None if len(image_local) == 0 else image_local[:viz_num],
                    'image': None if len(y_visual) == 0 else torch.zeros_like(y_visual0[:viz_num]),
                    'depth': None if len(depth_data) == 0 else depth_data[:viz_num],
                    'canny': None if len(canny_data) == 0 else canny_data[:viz_num],
                    'sketch': None if len(sketch_data) == 0 else sketch_data[:viz_num],
                    'masked': None if len(masked_video) == 0 else masked_video[:viz_num],
                    'motion': None if len(mv_data_video) == 0 else mv_data_video[:viz_num],
                    'single_sketch': None if len(single_sketch_data) == 0 else single_sketch_data[:viz_num],
                    'fps': fps[:viz_num]}
                ]
                    
                # Save generated videos 
                #---------------- txt + Motion -----------
                partial_keys_motion = ['y', 'motion']
                noise_motion = noise.clone()
                model_kwargs_motion = prepare_model_kwargs(partial_keys = partial_keys_motion,
                                        full_model_kwargs = full_model_kwargs,
                                        use_fps_condition = cfg.use_fps_condition)
                video_data_motion = diffusion.ddim_sample_loop(
                    noise=noise_motion,
                    model=model.eval(),
                    model_kwargs=model_kwargs_motion,
                    guide_scale=9.0,
                    ddim_timesteps=cfg.ddim_timesteps,
                    eta=0.0)
                
                visualize_with_model_kwargs(model_kwargs = model_kwargs_motion,
                    video_data = video_data_motion,
                    autoencoder = autoencoder,
                    ori_video = misc_backups,
                    viz_num = viz_num,
                    step = step,
                    caps = caps,
                    palette = palette,
                    cfg = cfg)
                #--------------------------------------

                #---------------- txt + Sketch --------
                partial_keys_1 = ['y', 'sketch']
                noise_1 = noise.clone()
                model_kwargs_1 = prepare_model_kwargs(partial_keys = partial_keys_1,
                                        full_model_kwargs = full_model_kwargs,
                                        use_fps_condition = cfg.use_fps_condition)
                video_data_1 = diffusion.ddim_sample_loop(
                    noise=noise_1,
                    model=model.eval(),
                    model_kwargs=model_kwargs_1,
                    guide_scale=9.0,
                    ddim_timesteps=cfg.ddim_timesteps,
                    eta=0.0)
                visualize_with_model_kwargs(model_kwargs = model_kwargs_1,
                    video_data = video_data_1,
                    autoencoder = autoencoder,
                    ori_video = misc_backups,
                    viz_num = viz_num,
                    step = step,
                    caps = caps,
                    palette = palette,
                    cfg = cfg)
                #--------------------------------------

                #---------------- txt + Depth --------
                partial_keys_2 = ['y', 'depth']
                noise_2 = noise.clone()
                model_kwargs_2 = prepare_model_kwargs(partial_keys = partial_keys_2,
                                        full_model_kwargs = full_model_kwargs,
                                        use_fps_condition = cfg.use_fps_condition)
                video_data_2 = diffusion.ddim_sample_loop(
                    noise=noise_2,
                    model=model.eval(),
                    model_kwargs=model_kwargs_2,
                    guide_scale=9.0,
                    ddim_timesteps=cfg.ddim_timesteps,
                    eta=0.0)
                visualize_with_model_kwargs(model_kwargs = model_kwargs_2,
                    video_data = video_data_2,
                    autoencoder = autoencoder,
                    ori_video = misc_backups,
                    viz_num = viz_num,
                    step = step,
                    caps = caps,
                    palette = palette,
                    cfg = cfg)
                #--------------------------------------

                #---------------- txt + local_image --------
                partial_keys_2_local_image = ['y', 'local_image']
                noise_2_local_image = noise.clone()
                model_kwargs_2_local_image = prepare_model_kwargs(partial_keys = partial_keys_2_local_image,
                                        full_model_kwargs = full_model_kwargs,
                                        use_fps_condition = cfg.use_fps_condition)
                video_data_2_local_image = diffusion.ddim_sample_loop(
                    noise=noise_2_local_image,
                    model=model.eval(),
                    model_kwargs=model_kwargs_2_local_image,
                    guide_scale=9.0,
                    ddim_timesteps=cfg.ddim_timesteps,
                    eta=0.0)
                visualize_with_model_kwargs(model_kwargs = model_kwargs_2_local_image,
                    video_data = video_data_2_local_image,
                    autoencoder = autoencoder,
                    ori_video = misc_backups,
                    viz_num = viz_num,
                    step = step,
                    caps = caps,
                    palette = palette,
                    cfg = cfg)
                #--------------------------------------
                
                #---------------- image + depth --------
                partial_keys_2_image = ['image', 'depth']
                noise_2_image = noise.clone()
                model_kwargs_2_image = prepare_model_kwargs(partial_keys = partial_keys_2_image,
                                        full_model_kwargs = full_model_kwargs,
                                        use_fps_condition = cfg.use_fps_condition)
                video_data_2_image = diffusion.ddim_sample_loop(
                    noise=noise_2_image,
                    model=model.eval(),
                    model_kwargs=model_kwargs_2_image,
                    guide_scale=9.0,
                    ddim_timesteps=cfg.ddim_timesteps,
                    eta=0.0)
                visualize_with_model_kwargs(model_kwargs = model_kwargs_2_image,
                    video_data = video_data_2_image,
                    autoencoder = autoencoder,
                    ori_video = misc_backups,
                    viz_num = viz_num,
                    step = step,
                    caps = caps,
                    palette = palette,
                    cfg = cfg)
                #--------------------------------------
                
                #---------------- text + mask --------
                partial_keys_3 = ['y', 'masked']
                noise_3 = noise.clone()
                model_kwargs_3 = prepare_model_kwargs(partial_keys = partial_keys_3,
                                        full_model_kwargs = full_model_kwargs,
                                        use_fps_condition = cfg.use_fps_condition)
                
                video_data_3 = diffusion.ddim_sample_loop(
                    noise=noise_3,
                    model=model.eval(),
                    model_kwargs=model_kwargs_3,
                    guide_scale=9.0,
                    ddim_timesteps=cfg.ddim_timesteps,
                    eta=0.0)
                visualize_with_model_kwargs(model_kwargs = model_kwargs_3,
                    video_data = video_data_3,
                    autoencoder = autoencoder,
                    ori_video = misc_backups,
                    viz_num = viz_num,
                    step = step,
                    caps = caps,
                    palette = palette,
                    cfg = cfg)
                #--------------------------------------

    if cfg.rank == 0:
        # send a sign to oss to indicate the training is completed
        logging.info('Congratulations! The inference is completed!')

    # synchronize to finish some processes
    if not cfg.debug:
        torch.cuda.synchronize()
        dist.barrier()

def prepare_model_kwargs(partial_keys, full_model_kwargs, use_fps_condition):
    for partial_key in partial_keys:
        assert partial_key in ['y', 'depth', 'canny', 'masked', 'sketch', "image", "motion", "local_image"]
    
    if use_fps_condition is True:
        partial_keys.append('fps')

    partial_model_kwargs = [{}, {}]
    for partial_key in partial_keys:
        partial_model_kwargs[0][partial_key] = full_model_kwargs[0][partial_key]
        partial_model_kwargs[1][partial_key] = full_model_kwargs[1][partial_key]

    return partial_model_kwargs


def visualize_with_model_kwargs(model_kwargs,
                                video_data,
                                autoencoder,
                                # ref_imgs,
                                ori_video,
                                viz_num,
                                step,
                                caps,
                                palette,
                                cfg):
    scale_factor = 0.18215
    video_data = 1. / scale_factor * video_data

    bs_vd = video_data.shape[0]
    video_data = rearrange(video_data, 'b c f h w -> (b f) c h w')
    chunk_size = min(16, video_data.shape[0])
    video_data_list = torch.chunk(video_data, video_data.shape[0]//chunk_size, dim=0)
    decode_data = []
    for vd_data in video_data_list:
        tmp = autoencoder.decode(vd_data)
        decode_data.append(tmp)
    video_data = torch.cat(decode_data,dim=0)
    video_data = rearrange(video_data, '(b f) c h w -> b c f h w', b = bs_vd)
    ori_video = ori_video[:viz_num]
    
    # upload conditional texts and videos
    oss_key_dir = osp.join(cfg.log_dir, f"step_{step}" + '-' + f"{'_'.join(model_kwargs[0].keys())}")
    # oss_key = osp.join(cfg.log_dir, f"step_{step}" + '-' + f"{'_'.join(model_kwargs[0].keys())}/rank_{cfg.world_size}-{cfg.rank}.gif")
    oss_key = os.path.join(oss_key_dir, f"rank_{cfg.world_size}-{cfg.rank}.gif")
    text_key = osp.join(cfg.log_dir, 'text_description.txt')
    if not os.path.exists(oss_key_dir):
        os.makedirs(oss_key_dir, exist_ok=True)
    
    # Save videos and text inputs.
    try:
        del model_kwargs[0][list(model_kwargs[0].keys())[0]]
        del model_kwargs[1][list(model_kwargs[1].keys())[0]]
        ops.save_video_multiple_conditions(oss_key, video_data, model_kwargs, ori_video, palette,
                                           cfg.mean, cfg.std, nrow=1)
        if cfg.rank == 0: 
            texts = '\n'.join(caps[:viz_num])
            open(text_key, 'w').writelines(texts)
    except Exception as e:
        logging.info(f'Save text or video error. {e}')
    
    logging.info(f'Save videos to {oss_key}')