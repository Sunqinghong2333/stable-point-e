import os
import random
from copy import copy
import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm.auto import tqdm
import pandas as pd


class ShapeNetCore(Dataset):

    def __init__(self, text_file, pcd_file, num_points=1024):
        self.texts_data = pd.read_csv(text_file, sep='\t', encoding='ISO-8859-1', header = None, names=['id','desc','cate','img'])
        self.num_points = num_points
        with open(pcd_file) as f:
            pcd_list = f.readlines()
        self.pcd_paths = {name.split('/')[-1].split('.')[0]:name for name in pcd_list}
        drop_list = []
        for i, tmp in enumerate(self.texts_data['id']):
            if tmp not in self.pcd_paths:
                drop_list.append(i)
        self.texts_data = self.texts_data.drop(drop_list).reset_index(drop=True)

    def __len__(self):
        return len(self.texts_data)
    
    def load_pcd(self, pc_id):
        pcd_path = self.pcd_paths[pc_id][:-1]
        data = np.load(pcd_path)
        coords = data["coords"]  #(4096, 3)
        rgbs = np.column_stack([data["R"], data["G"], data["B"]]) * 255 #(4096, 3)
        assert (
            coords.shape[0] == rgbs.shape[0]
        ), "coords and rgbs should have the same length"

        coords = (coords - np.min(coords) ) / ( np.max(coords) -  np.min(coords))
        coords = (coords-0.5)
        pc = np.column_stack([coords, rgbs])
        pc = torch.from_numpy(pc).float()
        if self.num_points < pc.shape[0]:
            tmp_idx = torch.randint(0, pc.shape[0], (self.num_points,)).long().view(-1)
            pc = pc[tmp_idx,...]

        return pc

    def __getitem__(self, idx):
        
        pc_id = self.texts_data.loc[idx]['id']
        description = self.texts_data.loc[idx]['desc']
        cate = self.texts_data.loc[idx]['cate']
        pc = self.load_pcd(pc_id)
        
        assert isinstance(description, str), pc_id
        assert isinstance(pc_id, str), pc_id
        if not isinstance(cate, str):
            cate = ''
        return {
                    'pointcloud': pc,
                    'cate': cate,
                    'id': pc_id,
                    'desc': description,
                }


