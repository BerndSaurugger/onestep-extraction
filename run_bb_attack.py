import json
import requests
import io
import base64
from PIL import Image, PngImagePlugin
import fire
import os
import numpy as np
import torchvision
import math
import shutil
import pandas as pd
import torch
from huggingface_hub import snapshot_download, hf_hub_download
from utils import processing_utils
from utils.functions import prompt_to_folder


@torch.no_grad()
def run_bb_attack(out_parquet_file, parquet_file=None,n_seeds=4,seed_offset=0,make_grid_every=0,outfolder='bb_attack_vis/',caption_offset=0,n_captions=-1,model='runwayml/stable-diffusion-v1-5',dl_parquet_repo='fraisdufour/sd-stuff',dl_parquet_name='membership_attack_top30k.parquet', compute_images=False, local_dir='.', verb=True):
    """
    Runs the blackbox attack against stable diffusion models. Exploits the fact that verbatim copies have strong edges when synthesizing with few timesteps versus non copied images which appear blurry.
    
    For more details see "A Reproducible Extraction of Training Images from Diffusion Models" https://arxiv.org/abs/2305.08694
    
    Parameters:
        - out_parquet_file (str): Output file path for saving the attack's scores.
        - parquet_file (str, optional): Input parquet containing captions for attack, if it is a local file. Captions should be within field 'caption' in the pandas dataframe.
        - n_seeds (int, optional): Number of random seeds per caption. Default is 4.
        - seed_offset (int, optional): Starting offset for random seed generation. Default is 0.
        - outfolder (str, optional): Output folder path for visualizing attack. Default is disabled.
        - caption_offset (int, optional): Start at a specified offset caption in parquet_file. Useful if you want to divide work amongst several gpus/nodes.
        - n_captions (int, optional): Number of captions to be used from the input parquet. If set to -1, all available captions will be used. Default is -1.
        - model (str, optional): Pre-trained stable diffusion model to be used for generating synthetic data. Default is 'runwayml/stable-diffusion-v1-5'. Use stabilityai/stable-diffusion-2-base for sdv2 models.
        - dl_parquet_repo (str, optional): Huggingface base repo for input parquet.
        - dl_parquet_name (str, optional): Name of parquet within huggingface repo. Default is 'membership_attack_top30k.parquet'.
        - compute_images (bool, optional): Whether to compute and save the generated images (for visualization). Default false.
        - local_dir (str, optional): Local directory path for temporary file storage. Default is '.'.
        - verb: Print current caption being synthesized.
    Returns:
        None
    
    Saves blackbox scores to out_parquet_file.
    """
    if parquet_file is not None:
        d = pd.read_parquet(parquet_file)
    else:
        print(f'downloading parquet from hub {dl_parquet_repo}/{dl_parquet_name}')
        hf_hub_download(repo_id=dl_parquet_repo, filename=dl_parquet_name, repo_type="dataset", local_dir='.')
        d = pd.read_parquet(dl_parquet_name)
 
    from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
    from custom_ksampler import StableDiffusionKDiffusionPipeline
    pipe = StableDiffusionKDiffusionPipeline.from_pretrained(model, torch_dtype=torch.float16) 
    pipe.set_scheduler("sample_heun")
    pipe = pipe.to("cuda")

    if n_captions > 0:
        last_caption = min(len(d["caption"]),caption_offset + n_captions)
    else:
        last_caption = len(d["caption"])
        
    captions = np.array(d["caption"])[caption_offset:last_caption]
    edge_overlap_scores = np.zeros((len(captions),))
    d = d.iloc[list(range(caption_offset,last_caption))]
    
    # synthesize all and save, using only one step and no guidance scale (BB attack)
    for ci,c in enumerate(captions):
        if verb:
            print(f'synthing caption {ci},{c}...')
        prompt=c
        outfolder_prompt = prompt_to_folder(prompt, 200)
        os.makedirs(f'{outfolder}{outfolder_prompt}',exist_ok=True)

        imgs_out = []
        img_group = []
        for seed in range(seed_offset, seed_offset + n_seeds):
            generator = torch.Generator("cuda").manual_seed(seed)
            
            image = pipe(prompt,num_inference_steps=1,generator=generator,use_karras_sigmas=True).images[0] 
            
            fn = f'{outfolder}{outfolder_prompt}/{seed:04d}.jpg'
            image.save(fn)
            tmp_img = np.array(image.resize((256,256), resample=Image.LANCZOS)).astype('float32')
            img_group += [tmp_img]
        
        score = processing_utils.get_edge_intersection_score(img_group)
        edge_overlap_scores[ci] = score
                                  
    d['edge_scores'] = edge_overlap_scores
    d.to_parquet(out_parquet_file)
    
if __name__ == '__main__':
    fire.Fire(run_bb_attack)