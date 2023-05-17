#!/usr/bin/env python
# coding: utf-8
import torch
from tqdm.auto import tqdm
from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.configs import MODEL_CONFIGS, model_from_config
import numpy as np
import random


seeds = [
    10,
    100,
]
prompts = [
    'table', 'chair', 'airplane', 'knife', 'train', 'can', 'skateboard', 'keyboard', 'cap', 'boeing', 'airplane: boeing', 'a red airplane', 'a grey airplane',
    'a skateboard with wheels', 'a white skateboard', 'a trash can', 'a green can', 'a black keyboard', 'a white keyboard'
]
ckpt = "./point_e_save/train-0516-222547/00320000.pth"
folder_vis = "./point_e_save/train-0516-222547/vis"


os.system(f"mkdir -p {folder_vis}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with torch.no_grad():
    print('creating base model...')
    base_name = 'base40M-textvec'
    base_model = model_from_config(MODEL_CONFIGS[base_name], device)
    base_model.eval()
    base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])

    print('loading local checkpoint...')
    
    state = torch.load(ckpt, map_location=device)['model']
    state2 = {}
    for k,v in state.items():
        k2 = k.replace("module.", "")
        state2[k2] = v
    base_model.load_state_dict(state2)
    sampler = PointCloudSampler(
        device=device,
        models=[base_model],
        diffusions=[base_diffusion],
        num_points=[1024],
        aux_channels=['R', 'G', 'B'],
        guidance_scale=[3.0],
        use_karras=[True],
        karras_steps=[64],
        sigma_min=[1e-3],
        sigma_max=[120],
        s_churn=[3],
        model_kwargs_key_filter=['texts'], # Do not condition the upsampler at all
    )

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        for prompt in prompts:
            samples = None
            for x in tqdm(sampler.sample_batch_progressive(batch_size=1, model_kwargs=dict(texts=[prompt]))):   
                samples = x

            pc = sampler.output_to_point_clouds(samples)[0]        
            prompt = prompt.replace(" ", "_")
            file_out = f"{folder_vis}/{prompt}_seed-{seed}_64steps.ply"
            with open(file_out, "wb") as writer:
                pc.write_ply(writer)

        
