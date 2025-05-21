import os
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb
import models, utils_training, datasets, utils
from ConfigModel import ConfigEnhancTrAISformer
from data_loader import load_datasets

from utils_training3 import setup_trainer
from models3 import EnALSModel

class MockConfig:
    # EnhancTrAISformer params
    lat_size=100; lon_size=100; sog_size=50; cog_size=72
    n_lat_embd=100; n_lon_embd=100; n_sog_embd=50; n_cog_embd=10
    n_embd=260 # Sum = 192
    max_seqlen=60 # Reduced for example
    n_head=4; n_layer=4; attn_pdrop=0.1; resid_pdrop=0.1; embd_pdrop=0.1
    partition_mode="uniform"; blur=False # Keep example simple

    anomaly_path_w=None#"/content/drive/MyDrive/ALSModels/model_GRU.pth"

    # EnALSModel params
    anomaly_latent_dim=100; anomaly_state_type="LSTM"
    anomaly_loss_weight=1.5; anomaly_threshold=0.0001


    # Optimizer params
    learning_rate=1e-4; weight_decay=0.01; betas=(0.9, 0.95)
def main():
    cf = ConfigEnhancTrAISformer()
    utils.set_seed(42)
    torch.pi = torch.acos(torch.zeros(1)).item() * 2

    # Initialize Weights & Biases
    wandb.init(project="eEnhance Traisformer", config=cf.__dict__)

    # Logging setup
    if not os.path.isdir(cf.savedir):
        os.makedirs(cf.savedir)
        print('======= Create directory to store trained models: ' + cf.savedir)
    else:
        print('======= Directory to store trained models: ' + cf.savedir)
    utils.new_log(cf.savedir, "log")

    # Load datasets
    aisdatasets, aisdls = load_datasets(cf)
    cf.final_tokens = 2 * len(aisdatasets["train"]) * cf.max_seqlen

    # Model setup
    # model =EnhancTrAISformer(cf, partition_model=None)
    config = MockConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Model Instantiation ---
    enmodel = EnALSModel(config)
    model=enmodel.to(device)
    enmodel.anomaly_model.train()

    trainer = setup_trainer(model, aisdatasets, aisdls, cf)

    # Training
    if cf.retrain:
        trainer.train()

    # Evaluation
    model.load_state_dict(torch.load(cf.ckpt_path))
    evaluate_model(model, aisdls, cf)

    wandb.finish()

def evaluate_model(model, aisdls, cf):
    v_ranges = torch.tensor([2, 3, 0, 0]).to(cf.device)
    v_roi_min = torch.tensor([model.lat_min, -7, 0, 0]).to(cf.device)
    max_seqlen = cf.init_seqlen + 6 * 4

    model.eval()
    l_min_errors, l_mean_errors, l_masks = [], [], []

    with torch.no_grad():
        for it, (seqs, masks, seqlens, mmsis, time_starts) in tqdm(enumerate(aisdls["test"]), total=len(aisdls["test"])):
            seqs_init = seqs[:, :cf.init_seqlen, :].to(cf.device)
            masks = masks[:, :max_seqlen].to(cf.device)
            batchsize = seqs.shape[0]
            error_ens = torch.zeros((batchsize, max_seqlen - cf.init_seqlen, cf.n_samples)).to(cf.device)

            for i_sample in range(cf.n_samples):
                preds = trainers.sample(model, seqs_init, max_seqlen - cf.init_seqlen, temperature=1.0, sample=True,
                                        sample_mode=cf.sample_mode, r_vicinity=cf.r_vicinity, top_k=cf.top_k)
                inputs = seqs[:, :max_seqlen, :].to(cf.device)
                input_coords = (inputs * v_ranges + v_roi_min) * torch.pi / 180
                pred_coords = (preds * v_ranges + v_roi_min) * torch.pi / 180
                d = utils.haversine(input_coords, pred_coords) * masks
                error_ens[:, :, i_sample] = d[:, cf.init_seqlen:]

            l_min_errors.append(error_ens.min(dim=-1))
            l_mean_errors.append(error_ens.mean(dim=-1))
            l_masks.append(masks[:, cf.init_seqlen:])

    plot_errors(l_min_errors, l_masks, cf)

def plot_errors(l_min_errors, l_masks, cf):
    l_min = [x.values for x in l_min_errors]
    m_masks = torch.cat(l_masks, dim=0)
    min_errors = torch.cat(l_min, dim=0) * m_masks
    pred_errors = min_errors.sum(dim=0) / m_masks.sum(dim=0)
    pred_errors = pred_errors.detach().cpu().numpy()

    plt.figure(figsize=(9, 6), dpi=150)
    v_times = np.arange(len(pred_errors)) / 6
    plt.plot(v_times, pred_errors)

    for i, timestep in enumerate([6, 12, 18]):
        plt.plot(i + 1, pred_errors[timestep], "o")
        plt.plot([i + 1, i + 1], [0, pred_errors[timestep]], "r")
        plt.plot([0, i + 1], [pred_errors[timestep], pred_errors[timestep]], "r")
        plt.text(i + 1.12, pred_errors[timestep] - 0.5, "{:.4f}".format(pred_errors[timestep]), fontsize=10)

    plt.xlabel("Time (hours)")
    plt.ylabel("Prediction errors (km)")
    plt.xlim([0, 12])
    plt.ylim([0, 20])

    # Log plot to wandb
    wandb.log({"Prediction Error Plot": wandb.Image(plt)})

    plt.savefig(cf.savedir + "prediction_error.png")

if __name__ == "__main__":
    main()
