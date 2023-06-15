import os.path as osp
import torch
import time
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as T
from collections import defaultdict
import re
import pickle
import json
import random
import numpy as np
from io import BytesIO
from PIL import Image
import artist.ops as ops
import cv2
from skimage.color import rgb2lab, lab2rgb
import datetime
# ADD
import os
from mvextractor.videocap import VideoCap
import subprocess
import binascii
from ipdb import set_trace
import imageio

import utils.logging as logging
logger = logging.get_logger(__name__)

def pre_caption(caption, max_words):
    caption = re.sub(
        r"([,.'!?\"()*#:;~])",
        '',
        caption.lower(),
    ).replace('-', ' ').replace('/', ' ').replace('<person>', 'person')

    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n')
    caption = caption.strip(' ')

    #truncate caption
    caption_words = caption.split(' ')
    if len(caption_words) > max_words:
        caption = ' '.join(caption_words[:max_words])

    return caption

def random_resize(img, size):
    return TF.resize(img, size, interpolation=random.choice([
        InterpolationMode.BILINEAR,
        InterpolationMode.BICUBIC,
        InterpolationMode.LANCZOS]))

def rand_name(length=16, suffix=''):
    name = binascii.b2a_hex(os.urandom(length)).decode('utf-8')
    if suffix:
        if not suffix.startswith('.'):
            suffix = '.' + suffix
        name += suffix
    return name

def draw_motion_vectors(frame, motion_vectors):
    if len(motion_vectors) > 0:
        num_mvs = np.shape(motion_vectors)[0]
        for mv in np.split(motion_vectors, num_mvs):
            start_pt = (mv[0, 3], mv[0, 4])
            end_pt = (mv[0, 5], mv[0, 6])
            cv2.arrowedLine(frame, start_pt, end_pt, (0, 0, 255), 1, cv2.LINE_AA, 0, 0.1)
            # cv2.arrowedLine(frame, start_pt, end_pt, (0, 0, 255), 2, cv2.LINE_AA, 0, 0.2)
    return frame

def extract_motion_vectors(input_video,fps=4, dump=False, verbose=False, visual_mv=False):

    if dump:
        now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        for child in ["frames", "motion_vectors"]:
            os.makedirs(os.path.join(f"out-{now}", child), exist_ok=True)
    temp = rand_name()
    # tmp_video = f'{temp}_{input_video}'
    tmp_video = os.path.join(input_video.split("/")[0], f'{temp}' +input_video.split("/")[-1])
    videocapture = cv2.VideoCapture(input_video)
    frames_num = videocapture.get(cv2.CAP_PROP_FRAME_COUNT)
    fps_video =videocapture.get(cv2.CAP_PROP_FPS)
    # check if enough frames
    if frames_num/fps_video*fps>16: #
        fps = max(fps, 1)
    else:
        fps = int(16/(frames_num/fps_video)) + 1
    ffmpeg_cmd = f'ffmpeg -threads 8 -loglevel error -i {input_video} -filter:v fps={fps} -c:v mpeg4 -f rawvideo {tmp_video}'

    if os.path.exists(tmp_video):
        os.remove(tmp_video)
    
    subprocess.run(args=ffmpeg_cmd,shell=True,timeout=120)

    cap = VideoCap()
    # open the video file
    ret = cap.open(tmp_video)
    if not ret:
        raise RuntimeError(f"Could not open {tmp_video}")
    
    step = 0
    times = []

    frame_types = []
    frames = []
    mvs = []
    mvs_visual = []
    # continuously read and display video frames and motion vectors
    while True:
        if verbose:
            print("Frame: ", step, end=" ")

        tstart = time.perf_counter()

        # read next video frame and corresponding motion vectors
        ret, frame, motion_vectors, frame_type, timestamp = cap.read()

        tend = time.perf_counter()
        telapsed = tend - tstart
        times.append(telapsed)

        # if there is an error reading the frame
        if not ret:
            if verbose:
                print("No frame read. Stopping.")
            break

        frame_save = np.zeros(frame.copy().shape, dtype=np.uint8) # *255
        if visual_mv:
            frame_save = draw_motion_vectors(frame_save, motion_vectors)

        # store motion vectors, frames, etc. in output directory
        dump = False
        if frame.shape[1] >= frame.shape[0]:
            w_half = (frame.shape[1] - frame.shape[0])//2
            if dump:
                cv2.imwrite(os.path.join(f"./mv_visual/", f"frame-{step}.jpg"), frame_save[:,w_half:-w_half])
            mvs_visual.append(frame_save[:,w_half:-w_half])
        else:
            h_half = (frame.shape[0] - frame.shape[1])//2
            if dump:
                cv2.imwrite(os.path.join(f"./mv_visual/", f"frame-{step}.jpg"), frame_save[h_half:-h_half,:])
            mvs_visual.append(frame_save[h_half:-h_half,:])

        h,w = frame.shape[:2]
        mv = np.zeros((h,w,2))
        position = motion_vectors[:,5:7].clip((0,0),(w-1,h-1))
        mv[position[:,1],position[:,0]]=motion_vectors[:,0:1]*motion_vectors[:,7:9]/motion_vectors[:, 9:]

        step += 1
        frame_types.append(frame_type)
        frames.append(frame)
        mvs.append(mv)
        # mvs_visual.append(frame_save)
    if verbose:
        print("average dt: ", np.mean(times))
    cap.release()

    if os.path.exists(tmp_video):
        os.remove(tmp_video)

    return frame_types,frames,mvs, mvs_visual


class VideoDataset(Dataset):
    def __init__(self, 
                cfg, 
                tokenizer=None,
                max_words=30,
                feature_framerate=1,
                max_frames=16,
                image_resolution=224,
                transforms=None,
                mv_transforms = None,
                misc_transforms = None,
                vit_transforms=None,
                vit_image_size = 336,
                misc_size = 384):

        self.cfg = cfg
        
        self.tokenizer = tokenizer
        self.max_words = max_words
        self.feature_framerate = feature_framerate
        self.max_frames = max_frames
        self.image_resolution = image_resolution
        self.transforms = transforms
        self.vit_transforms = vit_transforms
        self.vit_image_size = vit_image_size
        self.misc_transforms = misc_transforms
        self.misc_size = misc_size

        self.mv_transforms = mv_transforms

        self.video_cap_pairs = [[self.cfg.input_video, self.cfg.input_text_desc]]
        self.Vit_image_random_resize = T.Resize((vit_image_size, vit_image_size))
 
        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}

    def __len__(self):
        return len(self.video_cap_pairs)

    def __getitem__(self, index):

        video_key, cap_txt = self.video_cap_pairs[index]

        total_frames = None

        feature_framerate = self.feature_framerate
        if os.path.exists(video_key):
            try:
                ref_frame, vit_image, video_data, misc_data, mv_data = self._get_video_traindata(video_key, feature_framerate, total_frames, self.cfg.mvs_visual)
            except Exception as e:
                print('{} get frames failed... with error: {}'.format(video_key, e), flush=True)
                
                ref_frame = torch.zeros(3, self.vit_image_size, self.vit_image_size)
                vit_image = torch.zeros(3,self.vit_image_size,self.vit_image_size)
                video_data = torch.zeros(self.max_frames, 3, self.image_resolution, self.image_resolution)
                misc_data = torch.zeros(self.max_frames, 3, self.misc_size, self.misc_size)
                
                mv_data = torch.zeros(self.max_frames, 2, self.image_resolution, self.image_resolution)
        else:
            print("The video path does not exist or no video dir provided!")
            ref_frame = torch.zeros(3, self.vit_image_size, self.vit_image_size)
            vit_image = torch.zeros(3,self.vit_image_size,self.vit_image_size)
            video_data = torch.zeros(self.max_frames, 3, self.image_resolution, self.image_resolution)
            misc_data = torch.zeros(self.max_frames, 3, self.misc_size, self.misc_size)
            
            mv_data = torch.zeros(self.max_frames, 2, self.image_resolution, self.image_resolution)
            

        # inpainting mask
        p = random.random()
        if p < 0.7:
            mask = ops.make_irregular_mask(512, 512)
        elif p < 0.9:
            mask = ops.make_rectangle_mask(512, 512)
        else:
            mask = ops.make_uncrop(512, 512)
        mask = torch.from_numpy(cv2.resize(mask, (self.misc_size,self.misc_size), interpolation=cv2.INTER_NEAREST)).unsqueeze(0).float()

        mask = mask.unsqueeze(0).repeat_interleave(repeats=self.max_frames,dim=0)


        return ref_frame, cap_txt, video_data, misc_data, feature_framerate, mask, mv_data

    def _get_video_traindata(self, video_key, feature_framerate, total_frames, visual_mv):

        # folder_name = "cache_temp/"
        # filename = folder_name + osp.basename(video_key)
        # if not os.path.exists(folder_name):
        #     os.makedirs(folder_name, exist_ok=True)
        # oss_path = osp.join(self.root_dir, video_key)
        # bucket, oss_key = ops.parse_oss_url(oss_path)
        # ops.get_object_to_file(bucket, oss_key, filename)
        filename = video_key
        for _ in range(5):
            try:
                frame_types,frames,mvs, mvs_visual = extract_motion_vectors(input_video=filename,fps=feature_framerate, visual_mv=visual_mv)
                # os.remove(filename)
                break
            except Exception as e:
                print('{} read video frames and motion vectors failed with error: {}'.format(video_key, e), flush=True)

        total_frames = len(frame_types)
        start_indexs = np.where((np.array(frame_types)=='I') & (total_frames - np.arange(total_frames) >= self.max_frames))[0]
        start_index = np.random.choice(start_indexs)
        indices = np.arange(start_index, start_index+self.max_frames)

        # note frames are in BGR mode, need to trans to RGB mode
        frames = [Image.fromarray(frames[i][:, :, ::-1]) for i in indices]
        mvs = [torch.from_numpy(mvs[i].transpose((2,0,1))) for i in indices]
        mvs = torch.stack(mvs)
        # set_trace()
        # if mvs_visual != None:
        if visual_mv:
            # images = [(mvs_visual[i][:,:,::-1]*255).astype('uint8') for i in indices]
            images = [(mvs_visual[i][:,:,::-1]).astype('uint8') for i in indices]
            # images = [mvs_visual[i] for i in indices]
            # images = [(image.numpy()*255).astype('uint8') for image in images]
            path = self.cfg.log_dir + "/visual_mv/" + video_key.split("/")[-1] + ".gif"
            if not os.path.exists(self.cfg.log_dir + "/visual_mv/"):
                os.makedirs(self.cfg.log_dir + "/visual_mv/", exist_ok=True)
            print("save motion vectors visualization to :", path)
            imageio.mimwrite(path, images, fps=8)

        # mvs_visual = [torch.from_numpy(mvs_visual[i].transpose((2,0,1))) for i in indices]
        # mvs_visual = torch.stack(mvs_visual)
        # mvs_visual = self.mv_transforms(mvs_visual)

        have_frames = len(frames)>0
        middle_indix = int(len(frames)/2)
        if have_frames:
            ref_frame = frames[middle_indix]
            vit_image = self.vit_transforms(ref_frame)
            misc_imgs_np = self.misc_transforms[:2](frames)
            misc_imgs = self.misc_transforms[2:](misc_imgs_np)
            frames = self.transforms(frames)
            mvs = self.mv_transforms(mvs)
        else:
            # ref_frame = Image.fromarray(np.zeros((3, self.image_resolution, self.image_resolution)))
            vit_image = torch.zeros(3,self.vit_image_size,self.vit_image_size)

        video_data = torch.zeros(self.max_frames, 3, self.image_resolution, self.image_resolution)
        mv_data = torch.zeros(self.max_frames, 2, self.image_resolution, self.image_resolution)
        misc_data = torch.zeros(self.max_frames, 3, self.misc_size, self.misc_size)
        if have_frames:
            video_data[:len(frames), ...] = frames      # [[XX...],[...], ..., [0,0...], [], ...]
            misc_data[:len(frames), ...] = misc_imgs
            mv_data[:len(frames), ...] = mvs
        

        ref_frame = vit_image  

        del frames
        del misc_imgs
        del mvs

        return ref_frame, vit_image, video_data, misc_data, mv_data

