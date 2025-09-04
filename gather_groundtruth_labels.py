import fire
import os
from PIL import Image
import glob
import numpy as np
import pandas as pd
import utils.processing_utils as processing_utils
import utils.dl_utils as dl_utils
from huggingface_hub import snapshot_download
import re
from utils.functions import prompt_to_folder

def compute_masked_mses(templates,masks,imgs):
    mses = np.zeros((len(imgs),len(templates)))
    for imgi,img in enumerate(imgs):
        for ti,(t,m) in enumerate(zip(templates,masks)):
            mse = ((t*m - img*m)**2).sum()/(m.mean())
            mses[imgi,ti] = mse.item()
    return mses

def compute_pairwise_mses(imgs1,imgs2):
    mses = np.zeros((len(imgs1),len(imgs2)))
    for imgi,img1 in enumerate(imgs1):
        for ti,img2 in enumerate(imgs2):
            mse = ((img1 - img2)**2).sum()
            mses[imgi,ti] = mse.item()
    return mses

def get_templates_and_masks(template_folder='templates/'):
    template_parquet = pd.read_parquet(f'{template_folder}metadata.parquet')
    t_urls = np.array(template_parquet["url"])
    mask_files = list(template_parquet["mask_file"])
    template_files = list(template_parquet["img_file"])
    template_imgs,mask_imgs = [],[]
    for imgf,maskf in zip(template_files,mask_files):
        img,mask = [processing_utils.pil_img_to_torch(Image.open(file).resize((256,256))) for file in (imgf,maskf)]
        template_imgs += [img]
        mask_imgs += [mask]
    return template_imgs,mask_imgs,t_urls

def get_retrieved_imgs_and_urls(ret_folder='retrieved/'):
    md_ret = pd.read_parquet('retrieved/metadata.parquet')
    ret_imgs = []
    ret_urls = []
    for imgf,imgu in zip(md_ret["img_file"],md_ret["url"]):
        ret_imgs += [processing_utils.pil_img_to_torch(Image.open(imgf).resize((256,256)))]
        ret_urls += [imgu]
    return ret_imgs,np.array(ret_urls)

def get_files_from_path(folder_path,prefix,postfix):
    files = []
    for root, _, filenames in os.walk(folder_path):
        for filename in filenames:
            if filename.startswith(prefix) and filename.endswith(postfix):
                files.append(os.path.join(root, filename))
    files = sorted(files)
    return files

def gather_groundtruths(parquet_file='sdv1_wb_groundtruth.parquet',
                       out_parquet_file='test.parquet',
                       gen_folder='memb_top500_synthall/',
                       matching_real_folder = 'matched_and_real_images/matched/',
                       recompute_real_img_mse=True,
                       N_imgs_gen=-1,
                       n_imgs_template_thresh=0,
                       download_templates=True,
                       download_reals=True):

    if download_templates:
        print('downloading templates...')
        p = snapshot_download(repo_id="fraisdufour/templates-verbs", repo_type="dataset",local_dir='.')

    d = pd.read_parquet(parquet_file)

    if download_reals:
        print('downloading matching...')
        urls = list(d["url"])
        real_out = parquet_file[:(-1)*len('.parquet')] + '/'
        dl_utils.dl_urls_concurrent(urls,real_out,nthreads=16)

    ret_imgs,ret_urls = get_retrieved_imgs_and_urls()
    templates,masks,t_urls = get_templates_and_masks()

    real_files = sorted(glob.glob(real_out + '/*.jpg'))
    overfit_types = []
    retrieved_urls = []
    gen_seeds = []
    i = 0
    verb_thresh = float(2.5e3)
    real_mses = []
    template_indices=[[] for _ in range(len(d["caption"]))]

    for ci,c in enumerate(list(d["caption"])):
        pf = prompt_to_folder(c)  # das strippt jetzt HTML
        folder = f'{gen_folder}/{pf}/'
        files = get_files_from_path(folder,'','.jpg')

        real_file = real_files[ci]
        real_img = processing_utils.pil_img_to_torch(Image.open(real_file).resize((256,256)))

        # files = get_files_from_path(gen_folder + prompt_to_folder(c),'','.jpg')
        if N_imgs_gen > 0:
            files = files[:N_imgs_gen]

        imgs = [processing_utils.pil_img_to_torch(Image.open(file).resize((256,256))) for file in files]
        if len(imgs) == 0:
            print(f"Skipping sample {ci}: No generated images for caption: {c}")
            overfit_types += ['N']
            retrieved_urls += [[]]
            gen_seeds += [[]]
            real_mses += [np.nan]
            continue

        real_mse = min([((img-real_img)**2).sum() for img in imgs])
        real_mses += [real_mse]

        mses = compute_masked_mses(templates,masks,imgs)
        rd,cd = np.nonzero(mses < float(2.5e3))
        if len(rd) > n_imgs_template_thresh:
            t_inds = np.unique(cd.ravel())
            seeds = np.unique(rd.ravel())
            print(f'{c} found {len(t_inds)} templates with {len(seeds)} generating seeds...')
            print(seeds[:2].ravel())
            gen_seeds += [seeds.tolist()]
            retrieved_urls += [t_urls[t_inds].tolist()]
            template_indices[ci] = [t_inds.tolist()]
            overfit_types += ['TV']
        elif real_mse < float(2.5e3):
            print(c, "is verb")
            overfit_types += ['MV']
            retrieved_urls += [[]]
            gen_seeds += [[]]
        else:
            mses = compute_pairwise_mses(imgs,ret_imgs)
            all_seeds = np.arange(len(imgs))
            rd,cd = np.nonzero(mses < float(3e3))
            if len(rd):
                t_inds = np.unique(cd.ravel()).ravel()
                seeds = all_seeds[np.unique(rd.ravel())]
                urls_verb = ret_urls[t_inds]
                print(f'{c} found {len(t_inds)} retrieved images with {len(seeds)} generating seeds...')
                gen_seeds += [seeds.tolist()]
                retrieved_urls += [urls_verb.tolist()]
                overfit_types += ['RV']
            else:
                print(f'{c} none')
                overfit_types += ['N']
                retrieved_urls += [[]]
                gen_seeds += [[]]

    real_mses = np.array(real_mses)
    new_dict = {'overfit_type':overfit_types,'gen_seeds':gen_seeds,'retrieved_urls':retrieved_urls}
    d['overfit_type'] = overfit_types
    d['gen_seeds'] = gen_seeds
    d['retrieved_urls'] = retrieved_urls
    d['mse_real_gen'] = real_mses
    d['template_indices'] = template_indices
    d.to_parquet(out_parquet_file)

if __name__=='__main__':
    fire.Fire(gather_groundtruths)
