
import os
import math
import logging

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader
from torch.nn import functional as F
import utils
import wandb

logger = logging.getLogger(__name__)



class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1  # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6  # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9  # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = None
    num_workers = 0  # for DataLoader

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
import matplotlib.pyplot as plt
import numpy as np
import wandb

def plot_predictions(seqs, preds, seqlens, n_plots, init_seqlen=5, img_path="plot.png"):
    fig, ax = plt.subplots(figsize=(12, 8), dpi=150)

    # اختيار كولور ماب احترافي وراقي
    cmap = plt.get_cmap("viridis")  # ممكن تختار plasma, inferno, magma, etc.

    preds_np = preds.detach().cpu().numpy()
    inputs_np = seqs.detach().cpu().numpy()

    for idx in range(n_plots):
        fig2, ax2 = plt.subplots(figsize=(12, 8), dpi=250)
        color = cmap(idx / n_plots)

        try:
            seqlen = seqlens[idx].item()
        except:
            continue

        # إدخالات البداية
        ax.plot(inputs_np[idx][:init_seqlen, 1], inputs_np[idx][:init_seqlen, 0],
                color=color, linestyle="-", linewidth=3, alpha=0.9, label="Initial Path" if idx == 0 else "")
        ax2.plot(inputs_np[idx][:init_seqlen, 1], inputs_np[idx][:init_seqlen, 0],
                color=color, linestyle="-", linewidth=3, alpha=0.9, label="Initial Path" if idx == 0 else "")

        # نقاط البداية
        ax.scatter(inputs_np[idx][:init_seqlen, 1], inputs_np[idx][:init_seqlen, 0],
                   color=color, edgecolor="black", s=60, zorder=5, label="Input Points" if idx == 0 else "")
        ax2.scatter(inputs_np[idx][:init_seqlen, 1], inputs_np[idx][:init_seqlen, 0],
                   color=color, edgecolor="black", s=60, zorder=5, label="Input Points" if idx == 0 else "")

        # المسار الكامل الأصلي
        ax.plot(inputs_np[idx][:seqlen, 1], inputs_np[idx][:seqlen, 0],
                linestyle="--", linewidth=2, color=color, alpha=0.7, label="Full Input" if idx == 0 else "")
        ax2.plot(inputs_np[idx][:seqlen, 1], inputs_np[idx][:seqlen, 0],
                linestyle="--", linewidth=2, color=color, alpha=0.7, label="Full Input" if idx == 0 else "")

        # التوقعات
        ax.scatter(preds_np[idx][init_seqlen:, 1], preds_np[idx][init_seqlen:, 0],
                   marker="X", s=100, color=color, edgecolor="black", alpha=0.9, label="Predicted Points" if idx == 0 else "")
        ax2.scatter(preds_np[idx][init_seqlen:, 1], preds_np[idx][init_seqlen:, 0],
                   marker="X", s=100, color=color, edgecolor="black", alpha=0.9, label="Predicted Points" if idx == 0 else "")

        # تنسيق إضافي
        for axis in [ax2]:
            axis.set_xlim([-0.05, 1.05])
            axis.set_ylim([-0.05, 1.05])
            axis.set_xlabel("Longitude", fontsize=14, fontweight="bold")
            axis.set_ylabel("Latitude", fontsize=14, fontweight="bold")
            axis.set_title(f"Ship Trajectory Prediction {idx}", fontsize=18, fontweight="bold", color="navy")
            axis.legend(loc="best", fontsize=12, frameon=True, facecolor="white", edgecolor="gray")
            axis.grid(True, linestyle="--", linewidth=0.7, alpha=0.6)

        # رفع على W&B
        wandb.log({f"samples-{idx}": wandb.Image(fig2)})
        plt.close(fig2)  # إغلاق الشكل الفردي بعد رفعه لتوفير الذاكرة

    # الشكل الكبير النهائي
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel("Longitude", fontsize=16, fontweight="bold")
    ax.set_ylabel("Latitude", fontsize=16, fontweight="bold")
    ax.set_title("All Ship Trajectories", fontsize=20, fontweight="bold", color="darkred")
    ax.legend(loc="upper right", fontsize=13, frameon=True, facecolor="white", edgecolor="black")
    ax.grid(True, linestyle="--", linewidth=1.0, alpha=0.7)

    # حفظ الشكل
    plt.savefig(img_path, dpi=300, bbox_inches="tight")
    # plt.show()

import wandb
import matplotlib.pyplot as plt
import torch
import matplotlib.pyplot as plt
import wandb

def infer_and_evaluate_and_plot(model, x_context, x_input, sample=True, top_k=5):
    """
    Samples next step, converts indices to continuous, evaluates anomaly scores, and plots the results.

    Args:
        model: Trained model with sampling and scoring methods.
        x_context (Tensor): Context input for sampling.
        x_input (Tensor): Input for anomaly evaluation.
        sample (bool): Whether to sample or use argmax.
        top_k (int): Top-k sampling if applicable.

    Returns:
        dict: Results including predicted indices, continuous values, and anomaly scores.
    """

    results = {}
  ##  x_context=x_context.detach().cpu().numpy()

    # --- Sampling next step ---

    print("\n--- Inference: Predict Next Step ---")
    # context_len = 20
    # x_context = x_input[:, :context_len, :4]
    # print(f"Context shape: {x_context.shape}")
    model.eval()
    lat_ix, lon_ix, sog_ix, cog_ix = model.sample_next_step(x_context, sample=sample, top_k=top_k)

    next_indices = torch.cat([lat_ix, lon_ix, sog_ix, cog_ix], dim=-1)  # (batch, 4)
    next_continuous = model.indices_to_continuous(next_indices)
    print(f"Predicted next continuous values (approx): \n{next_continuous.tolist()}")

    results['next_indices'] = next_indices
    results['next_continuous'] = next_continuous

    # --- Anomaly on input ---
    print("\n--- Inference: Evaluate Anomaly ---")
    anomaly_scores = model.get_anomaly_scores(x_input, score_type='elbo')
    print(f"Anomaly Scores (ELBO) shape: {anomaly_scores.shape}, Mean: {anomaly_scores.mean().item():.4f}")

    results['anomaly_scores_input'] = anomaly_scores

    # --- Anomaly on generated point ---
    print("\n--- Inference: Evaluate Generated Anomaly (Stage 5 concept) ---")
    generated_point_for_eval = next_continuous.unsqueeze(1)  # (batch, 1, features)
    anomaly_scores_gen = model.get_anomaly_scores(generated_point_for_eval, score_type='elbo')

    results['anomaly_scores_generated'] = anomaly_scores_gen


    wandb.log({
        "Anomaly Score":anomaly_scores_gen.mean().item(),
        "Anomaly Score Input":anomaly_scores.mean().item()
    })

    plt.show()

    return results

def log_trajectory_to_wandb(seqs, preds, seqlens, init_seqlen=10, n_plots=5, name_prefix="Trajectory"):
    """
    Logs interactive ship trajectory plots to W&B.

    Args:
        seqs (Tensor): Input sequences (B, T, 2).
        preds (Tensor): Predicted sequences (B, T, 2).
        seqlens (Tensor): Sequence lengths.
        init_seqlen (int): Initial length to highlight.
        n_plots (int): Number of plots to log.
        name_prefix (str): Prefix for W&B plot names.
    """
    cmap = plt.cm.get_cmap("plasma")
    preds_np = preds.detach().cpu().numpy()
    inputs_np = seqs.detach().cpu().numpy()

    for idx in range(n_plots):
        try:
            seqlen = seqlens[idx].item()
        except:
            continue

        # تجهيز الداتا
        data = []
        for i in range(init_seqlen):
            data.append([inputs_np[idx][i, 1], inputs_np[idx][i, 0], "Initial"])
        for i in range(init_seqlen, seqlen):
            data.append([inputs_np[idx][i, 1], inputs_np[idx][i, 0], "Full Input"])
        for i in range(init_seqlen, preds_np.shape[1]):
            data.append([preds_np[idx][i, 1], preds_np[idx][i, 0], "Predicted"])

        table = wandb.Table(data=data, columns=["Longitude", "Latitude", "Type"])

        # رفع Scatter plot تفاعلي
        wandb.log({
            f"{name_prefix}-{idx}": wandb.plot.scatter(
                table,
                x="Longitude",
                y="Latitude",

                title=f"Ship Trajectory Prediction {idx}"
            )
        })

class Trainer:

    def __init__(self, model, train_dataset, test_dataset, config, savedir=None, device=torch.device("cpu"), aisdls={},
                 INIT_SEQLEN=0):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.savedir = savedir

        self.device = device
        self.model = model.model.to(device)
        self.desmodel=model
        self.aisdls = aisdls
        self.INIT_SEQLEN = INIT_SEQLEN

    def save_checkpoint(self, best_epoch):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        #         logging.info("saving %s", self.config.ckpt_path)
        logging.info(f"Best epoch: {best_epoch:03d}, saving model to {self.config.ckpt_path}")
        torch.save(raw_model.state_dict(), self.config.ckpt_path)

    def train(self):
        model, config, aisdls, INIT_SEQLEN, = self.model, self.config, self.aisdls, self.INIT_SEQLEN
        raw_model = model.module if hasattr(self.model, "module") else model
        desmodel = self.desmodel
        raw_model=raw_model
        optimizer = raw_model.configure_optimizers(config)
        if model.mode in ("gridcont_gridsin", "gridcont_gridsigmoid", "gridcont2_gridsigmoid",):
            return_loss_tuple = True
        else:
            return_loss_tuple = False

        def run_epoch(split, epoch=0):
            is_train = split == 'Training'
            model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            loader = DataLoader(data, shuffle=True, pin_memory=True,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers)

            losses = []
            n_batches = len(loader)
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            d_loss, d_reg_loss, d_n = 0, 0, 0
            for it, (seqs, masks, seqlens, mmsis, time_starts) in pbar:

                # place data on the correct device
                seqs = seqs.to(self.device)
                masks = masks[:, :-1].to(self.device)

                # forward the model
                with torch.set_grad_enabled(is_train):
                    if return_loss_tuple:
                        logits, loss, loss_tuple = desmodel(seqs,
                                                         masks=masks,
                                                         with_targets=True,
                                                         return_loss_tuple=return_loss_tuple)
                    else:
                        logits, loss = desmodel(seqs, masks=masks, with_targets=True)
                    loss = loss.mean()  # collapse all losses if they are scattered on multiple gpus
                    losses.append(loss.item())

                d_loss += loss.item() * seqs.shape[0]
                if return_loss_tuple:
                    reg_loss = loss_tuple[-1]
                    reg_loss = reg_loss.mean()
                    d_reg_loss += reg_loss.item() * seqs.shape[0]
                d_n += seqs.shape[0]
                if is_train:

                    # backprop and update the parameters
                    model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    optimizer.step()

                    # decay the learning rate based on our progress
                    if config.lr_decay:
                        self.tokens += (
                                seqs >= 0).sum()  # number of tokens processed this step (i.e. label is not -100)
                        if self.tokens < config.warmup_tokens:
                            # linear warmup
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            # cosine learning rate decay
                            progress = float(self.tokens - config.warmup_tokens) / float(
                                max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate

                    # report progress
                    pbar.set_description(f"epoch {epoch + 1} iter {it}: loss {loss.item():.5f}. lr {lr:e}")

                    # tb logging
                    if config.tb_log:
                        tb.add_scalar("loss",
                                      loss.item(),
                                      epoch * n_batches + it)
                        tb.add_scalar("lr",
                                      lr,
                                      epoch * n_batches + it)

                        for name, params in model.head.named_parameters():
                            tb.add_histogram(f"head.{name}", params, epoch * n_batches + it)
                            tb.add_histogram(f"head.{name}.grad", params.grad, epoch * n_batches + it)
                        if model.mode in ("gridcont_real",):
                            for name, params in model.res_pred.named_parameters():
                                tb.add_histogram(f"res_pred.{name}", params, epoch * n_batches + it)
                                tb.add_histogram(f"res_pred.{name}.grad", params.grad, epoch * n_batches + it)

            if is_train:
                wandb.log({f"{split}loss": d_loss / d_n, "lr": lr,"degloss":d_reg_loss / d_n})
                if return_loss_tuple:
                    logging.info(
                        f"{split}, epoch {epoch + 1}, loss {d_loss / d_n:.5f}, {d_reg_loss / d_n:.5f}, lr {lr:e}.")
                else:
                    logging.info(f"{split}, epoch {epoch + 1}, loss {d_loss / d_n:.5f}, lr {lr:e}.")
            else:
                wandb.log({f"{split}loss": d_loss / d_n})
                if return_loss_tuple:
                    logging.info(f"{split}, epoch {epoch + 1}, loss {d_loss / d_n:.5f}.")
                else:
                    logging.info(f"{split}, epoch {epoch + 1}, loss {d_loss / d_n:.5f}.")

            if not is_train:
                test_loss = float(np.mean(losses))
                #                 logging.info("test loss: %f", test_loss)
                return test_loss

        best_loss = float('inf')
        self.tokens = 0  # counter used for learning rate decay
        best_epoch = 0

        for epoch in range(config.max_epochs):

            run_epoch('Training', epoch=epoch)
            if self.test_dataset is not None:
                test_loss = run_epoch('Valid', epoch=epoch)

            # supports early stopping based on the test loss, or just save always if no test set is provided
            good_model = self.test_dataset is None or test_loss < best_loss
            if self.config.ckpt_path is not None and good_model:
                best_loss = test_loss
                best_epoch = epoch
                self.save_checkpoint(best_epoch + 1)

            ## SAMPLE AND PLOT
            # ==========================================================================================
            # ==========================================================================================
            raw_model = model.module if hasattr(self.model, "module") else model
            # seqs, masks, seqlens, mmsis, time_starts = iter(aisdls["test"]).next()
            seqs, masks, seqlens, mmsis, time_starts = next(iter(aisdls["test"]))
            n_plots = 7
            init_seqlen = INIT_SEQLEN
            seqs_init = seqs[:n_plots, :init_seqlen, :].to(self.device)
            preds = sample(raw_model,
                           seqs_init,
                           96 - init_seqlen,
                           temperature=1.0,
                           sample=True,
                           sample_mode=self.config.sample_mode,
                           r_vicinity=self.config.r_vicinity,
                           top_k=self.config.top_k)


            infer_and_evaluate_and_plot(desmodel,seqs_init,seqs_init)

            img_path = os.path.join(self.savedir, f'epoch_{epoch + 1:03d}.jpg')



            plot_predictions(seqs, preds, seqlens, n_plots, init_seqlen=init_seqlen, img_path=img_path)
            log_trajectory_to_wandb(seqs, preds, seqlens, init_seqlen=init_seqlen, n_plots=n_plots, name_prefix="Trajectory")
            # infer_and_evaluate_and_plot(self.desmodel,preds.detach(),preds.detach())
            # تسجيلها في Weights & Biases
            #if epoch % 5 == 0:


            plt.close()

        # Final state
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        #         logging.info("saving %s", self.config.ckpt_path)
        logging.info(f"Last epoch: {epoch:03d}, saving model to {self.config.ckpt_path}")
        save_path = self.config.ckpt_path.replace("model.pt", f"model_{epoch + 1:03d}.pt")
        torch.save(raw_model.state_dict(), save_path)


@torch.no_grad()
def sample(model,
           seqs,
           steps,
           temperature=1.0,
           sample=False,
           sample_mode="pos_vicinity",
           r_vicinity=20,
           top_k=None):

    max_seqlen = model.get_max_seqlen()
    model.eval()
    for k in range(steps):
        seqs_cond = seqs if seqs.size(1) <= max_seqlen else seqs[:, -max_seqlen:]  # crop context if needed

        # logits.shape: (batch_size, seq_len, data_size)
        logits, _ = model(seqs_cond)
        d2inf_pred = torch.zeros((logits.shape[0], 4)).to(seqs.device) + 0.5

        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature  # (batch_size, data_size)

        lat_logits, lon_logits, sog_logits, cog_logits = \
            torch.split(logits, (model.lat_size, model.lon_size, model.sog_size, model.cog_size), dim=-1)

        # optionally crop probabilities to only the top k options
        if sample_mode in ("pos_vicinity",):
            idxs, idxs_uniform = model.to_indexes(seqs_cond[:, -1:, :])
            lat_idxs, lon_idxs = idxs_uniform[:, 0, 0:1], idxs_uniform[:, 0, 1:2]
            lat_logits = utils.top_k_nearest_idx(lat_logits, lat_idxs, r_vicinity)
            lon_logits = utils.top_k_nearest_idx(lon_logits, lon_idxs, r_vicinity)

        if top_k is not None:
            lat_logits = utils.top_k_logits(lat_logits, top_k)
            lon_logits = utils.top_k_logits(lon_logits, top_k)
            sog_logits = utils.top_k_logits(sog_logits, top_k)
            cog_logits = utils.top_k_logits(cog_logits, top_k)

        # apply softmax to convert to probabilities
        lat_probs = F.softmax(lat_logits, dim=-1)
        lon_probs = F.softmax(lon_logits, dim=-1)
        sog_probs = F.softmax(sog_logits, dim=-1)
        cog_probs = F.softmax(cog_logits, dim=-1)

        # sample from the distribution or take the most likely
        if sample:
            lat_ix = torch.multinomial(lat_probs, num_samples=1)  # (batch_size, 1)
            lon_ix = torch.multinomial(lon_probs, num_samples=1)
            sog_ix = torch.multinomial(sog_probs, num_samples=1)
            cog_ix = torch.multinomial(cog_probs, num_samples=1)
        else:
            _, lat_ix = torch.topk(lat_probs, k=1, dim=-1)
            _, lon_ix = torch.topk(lon_probs, k=1, dim=-1)
            _, sog_ix = torch.topk(sog_probs, k=1, dim=-1)
            _, cog_ix = torch.topk(cog_probs, k=1, dim=-1)

        ix = torch.cat((lat_ix, lon_ix, sog_ix, cog_ix), dim=-1)
        # convert to x (range: [0,1))
        x_sample = (ix.float() + d2inf_pred) / model.att_sizes

        # append to the sequence and continue
        seqs = torch.cat((seqs, x_sample.unsqueeze(1)), dim=1)

    return seqs



def setup_trainer(model, aisdatasets, aisdls, cf):
    return Trainer(model, aisdatasets["train"], aisdatasets["valid"], cf, savedir=cf.savedir, device=cf.device, aisdls=aisdls, INIT_SEQLEN=cf.init_seqlen)
