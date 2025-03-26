import os
import pickle
import numpy as np
from torch.utils.data import DataLoader
import datasets

def load_datasets(cf,moving_threshold = 0.005):
    
    l_pkl_filenames = [cf.trainset_name, cf.validset_name, cf.testset_name]
    Data, aisdatasets, aisdls = {}, {}, {}
    
    for phase, filename in zip(("train", "valid", "test"), l_pkl_filenames):
        datapath = os.path.join(cf.datadir, filename)
        with open(datapath, "rb") as f:
            l_pred_errors = pickle.load(f)
        
        for V in l_pred_errors:
            try:
                moving_idx = np.where(V["traj"][:, 2] > moving_threshold)[0][0]
            except:
                moving_idx = len(V["traj"]) - 1
            V["traj"] = V["traj"][moving_idx:, :]
        
        Data[phase] = [x for x in l_pred_errors if not np.isnan(x["traj"]).any() and len(x["traj"]) > cf.min_seqlen]
        
        dataset_class = datasets.AISDataset_grad if cf.mode in ("pos_grad", "grad") else datasets.AISDataset
        aisdatasets[phase] = dataset_class(Data[phase], max_seqlen=cf.max_seqlen + 1, device=cf.device)
        shuffle = phase != "test"
        aisdls[phase] = DataLoader(aisdatasets[phase], batch_size=cf.batch_size, shuffle=shuffle)
    # aisdls["valid"]=aisdls["train"]
    return aisdatasets, aisdls