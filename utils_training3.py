
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

def create_optimizer_des(model, lr=0.001, beta0=0.8, beta1=0.999, eps=1e-08, weight_decay=0.0):
    """
    Creates an AdamW optimizer with the specified parameters.

    Args:
        model (torch.nn.Module): The model to optimize.
        lr (float): Learning rate.
        beta0 (float): Beta1 value for the Adam optimizer.
        beta1 (float): Beta2 value for the Adam optimizer.
        eps (float): Term added to the denominator to improve numerical stability.
        weight_decay (float): Weight decay (L2 penalty).

    Returns:
        torch.optim.Optimizer: Configured AdamW optimizer.
    """
    return torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(beta0, beta1),
        eps=eps,
        weight_decay=weight_decay
    )


class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 32
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
    num_workers = 4 # for DataLoader

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

    model.eval()

    lat_ix, lon_ix, sog_ix, cog_ix = model.sample_next_step(x_context, sample=sample, top_k=top_k)

    next_indices = torch.cat([lat_ix, lon_ix, sog_ix, cog_ix], dim=-1)  # (batch, 4)
    next_continuous = model.indices_to_continuous(next_indices)

    results['next_indices'] = next_indices
    results['next_continuous'] = next_continuous

    # --- Anomaly on input ---
    anomaly_scores = model.anomaly_model.calculate_anomaly_rate(x_input)

    results['anomaly_scores_input'] = anomaly_scores

    # --- Anomaly on generated point ---
    generated_point_for_eval = next_continuous.unsqueeze(1)  # (batch, 1, features)
    anomaly_scores_gen = model.anomaly_model.calculate_anomaly_rate(generated_point_for_eval)

    results['anomaly_scores_generated'] = anomaly_scores_gen


    wandb.log({
        "Anomaly Score":anomaly_scores_gen,
        "Anomaly Score Input":anomaly_scores
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
        # Ensure the main model is on the correct device
        self.model = model.model.to(device)
        # Ensure the combined model wrapper is also on the correct device
        self.desmodel=model.to(device)
        self.aisdls = aisdls
        self.INIT_SEQLEN = INIT_SEQLEN
        # Initialize the anomaly model optimizer with parameters from the anomaly_model part
        self.optimizer_des = create_optimizer_des(self.desmodel.anomaly_model)

    def save_checkpoint(self, best_epoch):
        # DataParallel wrappers keep raw model object in .module attribute
        # Save the state_dict of the main model, not the combined one unless needed
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        logging.info(f"Best epoch: {best_epoch:03d}, saving main model to {self.config.ckpt_path}")
        torch.save(raw_model.state_dict(), self.config.ckpt_path)

        # Optionally save the anomaly model's state_dict separately
        # raw_anomaly_model = self.desmodel.anomaly_model.module if hasattr(self.desmodel.anomaly_model, "module") else self.desmodel.anomaly_model
        # anomaly_ckpt_path = self.config.ckpt_path.replace(".pt", "_anomaly.pt")
        # logging.info(f"Best epoch: {best_epoch:03d}, saving anomaly model to {anomaly_ckpt_path}")
        # torch.save(raw_anomaly_model.state_dict(), anomaly_ckpt_path)


    def train(self):
        model, config, aisdls, INIT_SEQLEN, = self.model, self.config, self.aisdls, self.INIT_SEQLEN
        # Use the raw model for the main optimizer configuration
        raw_model = model.module if hasattr(self.model, "module") else model
        desmodel = self.desmodel
        optimizer_des = self.optimizer_des
        # Configure optimizer for the main model
        optimizer = raw_model.configure_optimizers(config)

        if model.mode in ("gridcont_gridsin", "gridcont_gridsigmoid", "gridcont2_gridsigmoid",):
            return_loss_tuple = True
        else:
            return_loss_tuple = False

        def run_epoch(split, epoch=0):
            is_train = split == 'Training'
            # Set train/eval mode for both models
            model.train(is_train)
            desmodel.anomaly_model.train(is_train)

            data = self.train_dataset if is_train else self.test_dataset
            loader = DataLoader(data, shuffle=True, pin_memory=True,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers)

            losses = []
            n_batches = len(loader)
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            d_loss, d_reg_loss, d_n = 0, 0, 0

            # Variables to track anomaly losses specifically for logging/display
            total_anomaly_loss = 0
            total_anomaly_recon_loss = 0
            total_anomaly_kl_div = 0
            anomaly_batch_count = 0


            for it, (seqs, masks, seqlens, mmsis, time_starts) in pbar:

                # place data on the correct device
                seqs = seqs.to(self.device)
                # Masks for the Transformer loss (targets are seqs[:, 1:])
                # If the input sequence has length T, the targets are indices 1 to T-1,
                # so the mask should be (batch, T-1). Original code `masks[:, :-1]` is correct.
                masks = masks[:, :-1].to(self.device)


                # forward the model
                # Gradient calculation is enabled only if is_train is True
                with torch.set_grad_enabled(is_train):
                    # The desmodel forward returns total_loss, anomaly_loss, and loss_components
                    # total_loss = transformer_loss + anomaly_loss_weight * anomaly_loss
                    # anomaly_loss = anomaly_recon_loss + beta * anomaly_kl_div
                    if return_loss_tuple:
                        # When with_targets=True, desmodel calculates losses and returns them
                        logits, total_loss, anomaly_loss, loss_tuple = desmodel(seqs,
                                                                                masks=masks,
                                                                                with_targets=True,
                                                                                return_loss_tuple=return_loss_tuple)
                        # Unpack anomaly loss components if return_loss_tuple is True AND anomaly loss was computed
                        if loss_tuple[1] is not None: # Check if anomaly_loss was computed
                            _, _, anomaly_recon_loss, anomaly_kl_div, _, _, _, _ = loss_tuple
                        else: # If anomaly loss was not computed (e.g., sequence too short)
                            anomaly_recon_loss = torch.tensor(0.0, device=self.device)
                            anomaly_kl_div = torch.tensor(0.0, device=self.device)


                    else: # If return_loss_tuple is False, anomaly loss components are not explicitly returned
                         # In this case, anomaly_loss itself is the combined ELBO
                         logits, total_loss, anomaly_loss = desmodel(seqs, masks=masks, with_targets=True)
                         # We don't have access to recon and kl separately here unless we modify desmodel
                         # For simplicity, we'll just log the total anomaly_loss
                         anomaly_recon_loss = torch.tensor(0.0, device=self.device) # Placeholder
                         anomaly_kl_div = torch.tensor(0.0, device=self.device) # Placeholder


                    # Ensure total_loss is computed (it might be None if transformer_loss is None)
                    # This shouldn't happen with with_targets=True, but good practice.
                    # If both transformer and anomaly losses are None, total_loss would be None.
                    # Let's assume total_loss is always returned when with_targets=True based on the logic.
                    total_loss = total_loss.mean()  # collapse all losses if they are scattered on multiple gpus
                    losses.append(total_loss.item()) # Log the combined total loss

                    # Accumulate anomaly loss components for logging
                    if anomaly_loss is not None:
                        total_anomaly_loss += anomaly_loss.mean().item() * seqs.shape[0] # Use mean() if anomaly_loss is per batch item
                        total_anomaly_recon_loss += anomaly_recon_loss.mean().item() * seqs.shape[0] # Assuming recon/kl are also per batch item
                        total_anomaly_kl_div += anomaly_kl_div.mean().item() * seqs.shape[0]
                        anomaly_batch_count += seqs.shape[0] # Count batches contributing to anomaly loss


                d_loss += total_loss.item() * seqs.shape[0]
                if return_loss_tuple:

                    transf_loss, anomaly_l, anomaly_r, anomaly_k, t_lat, t_lon, t_sog, t_cog = loss_tuple
                    # Log individual transformer losses if needed
                    # For now, logging combined total_loss and individual anomaly components


                d_n += seqs.shape[0] # Total number of samples processed in this epoch

                if is_train:
                    # --- Anomaly Model Training Step ---
                    # ONLY backpropagate and update the anomaly model if it contributed to the loss
                    if anomaly_loss is not None:
                         optimizer_des.zero_grad()
                         # Backpropagate the anomaly loss. It's already meaned in the desmodel.forward return.
                         anomaly_loss.backward()
                         torch.nn.utils.clip_grad_norm_(desmodel.anomaly_model.parameters(), config.grad_norm_clip)
                         optimizer_des.step()


                    # --- Main Transformer Model Training Step ---
                    # The total_loss includes transformer_loss and weighted anomaly_loss
                    model.zero_grad()
                    total_loss.backward() # Backpropagate the combined total loss
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)

                    optimizer.step()


                    # decay the learning rate based on our progress
                    if config.lr_decay:
                        # Calculate tokens for LR decay based on sequence length and batch size
                        self.tokens += seqs.size(0) * seqs.size(1) # Number of tokens = Batch Size * Sequence Length
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
                    # Update pbar description to include anomaly loss
                    if anomaly_batch_count > 0:
                        pbar.set_description(f"epoch {epoch + 1} iter {it}: total loss {total_loss.item():.5f}, anomaly loss {total_anomaly_loss / anomaly_batch_count:.5f}. lr {lr:e}")
                    else:
                         pbar.set_description(f"epoch {epoch + 1} iter {it}: total loss {total_loss.item():.5f}. lr {lr:e}")


                    # tb logging (if config.tb_log is enabled and imported)
                    if hasattr(config, 'tb_log') and config.tb_log:
                        # Make sure 'tb' is imported or defined (e.g., from torch.utils.tensorboard import SummaryWriter)
                        # tb.add_scalar("loss/total_train", total_loss.item(), epoch * n_batches + it)
                        # if anomaly_loss is not None:
                        #     tb.add_scalar("loss/anomaly_train", anomaly_loss.mean().item(), epoch * n_batches + it) # Use mean if anomaly_loss is per batch item
                        #     if return_loss_tuple:
                        #         tb.add_scalar("loss/anomaly_recon_train", anomaly_recon_loss.mean().item(), epoch * n_batches + it)
                        #         tb.add_scalar("loss/anomaly_kl_train", anomaly_kl_div.mean().item(), epoch * n_batches + it)
                        # tb.add_scalar("lr", lr, epoch * n_batches + it)
                        pass # Assuming tensorboard logging is handled elsewhere or not strictly needed here

            # End of epoch logging
            if anomaly_batch_count > 0:
                 avg_anomaly_loss = total_anomaly_loss / anomaly_batch_count
                 avg_anomaly_recon_loss = total_anomaly_recon_loss / anomaly_batch_count
                 avg_anomaly_kl_div = total_anomaly_kl_div / anomaly_batch_count
            else:
                 avg_anomaly_loss = 0.0
                 avg_anomaly_recon_loss = 0.0
                 avg_anomaly_kl_div = 0.0


            if is_train:
                # Log combined loss, anomaly loss components, and learning rate
                wandb.log({f"{split}loss/total": d_loss / d_n,
                           f"{split}loss/anomaly_total": avg_anomaly_loss,
                           f"{split}loss/anomaly_recon": avg_anomaly_recon_loss,
                           f"{split}loss/anomaly_kl": avg_anomaly_kl_div,
                           "lr": lr})
                # Removed d_reg_loss logging as its meaning was unclear
                logging.info(
                    f"{split}, epoch {epoch + 1}, total loss {d_loss / d_n:.5f}, anomaly loss {avg_anomaly_loss:.5f}, lr {lr:e}.")
            else: # Validation split
                # Log combined loss and anomaly loss components for validation
                 wandb.log({f"{split}loss/total": d_loss / d_n,
                            f"{split}loss/anomaly_total": avg_anomaly_loss,
                            f"{split}loss/anomaly_recon": avg_anomaly_recon_loss,
                            f"{split}loss/anomaly_kl": avg_anomaly_kl_div})

                 logging.info(f"{split}, epoch {epoch + 1}, total loss {d_loss / d_n:.5f}, anomaly loss {avg_anomaly_loss:.5f}.")


            if not is_train:
                # Return the average total loss for early stopping
                test_loss = float(np.mean(losses)) # Mean of batch total losses
                return test_loss # Return total loss for validation

        best_loss = float('inf')
        self.tokens = 0  # counter used for learning rate decay
        best_epoch = 0

        for epoch in range(config.max_epochs):

            # Run training epoch
            run_epoch('Training', epoch=epoch)

            # Run validation epoch if test_dataset exists
            if self.test_dataset is not None:
                test_loss = run_epoch('Valid', epoch=epoch)

            # supports early stopping based on the test loss, or just save always if no test set is provided
            # Use the total test loss for deciding whether to save the model
            good_model = self.test_dataset is None or (self.test_dataset is not None and test_loss < best_loss)
            if self.config.ckpt_path is not None and good_model:
                best_loss = test_loss if self.test_dataset is not None else float('inf') # Update best_loss only if test_dataset exists
                best_epoch = epoch
                # Save the best model based on validation loss
                self.save_checkpoint(best_epoch + 1)

            ## SAMPLE AND PLOT
            # ==========================================================================================
            # ==========================================================================================
            # Get the raw transformer model for sampling (EnALSModel wraps EnhancTrAISformer)
            raw_model = model.module if hasattr(self.model, "module") else model
            # seqs, masks, seqlens, mmsis, time_starts = iter(aisdls["test"]).next() # Older way
            # Use next(iter(...)) for modern Python 3+
            seqs, masks, seqlens, mmsis, time_starts = next(iter(aisdls["test"]))

            # Move data to device for plotting and sampling
            seqs = seqs.to(self.device)
            seqlens = seqlens.to(self.device)


            n_plots = 7
            init_seqlen = INIT_SEQLEN # This comes from config.init_seqlen
            # Take the initial context for sampling
            seqs_init = seqs[:n_plots, :init_seqlen, :].to(self.device)

            # Sample the future sequence using the raw transformer model
            # Ensure the sample function uses the raw_model
            preds = sample(raw_model, # Pass the raw transformer model
                           seqs_init,
                           96 - init_seqlen, # Number of steps to predict
                           temperature=1.0,
                           sample=True,
                           sample_mode=self.config.sample_mode,
                           r_vicinity=self.config.r_vicinity,
                           top_k=self.config.top_k)

            # Perform inference and evaluate anomaly scores on the original input and predicted sequence
            # Pass the combined model (desmodel) for anomaly evaluation
            infer_and_evaluate_and_plot(desmodel, seqs_init, seqs) # Evaluate anomaly on the full original sequence

            # Plot predictions - plot_predictions expects full original seqs and predicted seqs
            # Make sure preds is on CPU for plotting if necessary, though matplotlib can often handle GPU tensors now.
            # It's safer to move to CPU for plotting functions that might use numpy.
            plot_predictions(seqs.cpu(), preds.cpu(), seqlens.cpu(), n_plots, init_seqlen=init_seqlen, img_path=os.path.join(self.savedir, f'epoch_{epoch + 1:03d}.jpg'))

            # Log trajectory plots to W&B
            log_trajectory_to_wandb(seqs[:,:,:2].cpu(), preds[:,:,:2].cpu(), seqlens.cpu(), init_seqlen=init_seqlen, n_plots=n_plots, name_prefix="Trajectory")

            plt.close('all') # Close all plot figures to free memory


        # Final state saving after training finishes
        # Get the raw model state_dict
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        logging.info(f"Training finished after {config.max_epochs} epochs. Saving final model to {self.config.ckpt_path}")
        # Save the final model state dictionary
        torch.save(raw_model.state_dict(), self.config.ckpt_path)

        # Optionally save the final anomaly model state_dict
        # raw_anomaly_model = self.desmodel.anomaly_model.module if hasattr(self.desmodel.anomaly_model, "module") else self.desmodel.anomaly_model
        # anomaly_final_ckpt_path = self.config.ckpt_path.replace(".pt", "_anomaly_final.pt")
        # logging.info(f"Saving final anomaly model to {anomaly_final_ckpt_path}")
        # torch.save(raw_anomaly_model.state_dict(), anomaly_final_ckpt_path)


@torch.no_grad()
def sample(model,
           seqs, # This should be the raw Transformer model (EnhancTrAISformer), not EnALSModel
           steps,
           temperature=1.0,
           sample=False,
           sample_mode="pos_vicinity",
           r_vicinity=20,
           top_k=None):
    """
    Generates future steps using the EnhancTrAISformer model.

    Args:
        model (EnhancTrAISformer): The raw transformer model for prediction.
        seqs (Tensor): Input context sequence (batch, seqlen, 4), normalized [0,1).
                       Should be on the correct device.
        steps (int): Number of future steps to predict.
        temperature (float): Softmax temperature for sampling.
        sample (bool): If True, sample; if False, take argmax.
        sample_mode (str): Sampling mode ("pos_vicinity", etc.).
        r_vicinity (int): Radius for vicinity sampling.
        top_k (int, optional): If set, restrict sampling to the top k.

    Returns:
        Tensor: Generated sequence including the initial context and predicted steps.
                Shape: (batch, initial_seqlen + steps, 4). On the same device as input seqs.
    """

    max_seqlen = model.get_max_seqlen()
    model.eval()
    generated_seqs = seqs.clone() # Start with the initial context

    for k in range(steps):
        # Only use the last `max_seqlen` tokens as context for prediction
        seqs_cond = generated_seqs if generated_seqs.size(1) <= max_seqlen else generated_seqs[:, -max_seqlen:]

        # Get logits from the transformer model
        # The raw transformer model only returns logits and potentially other things,
        # but the first return value is the logits (batch_size, seq_len, data_size).
        # We only need the logits for the *last* token in the context sequence.
        logits, *_ = model(seqs_cond) # Use the raw transformer model here
        logits = logits[:, -1, :] / temperature  # (batch_size, data_size)

        # Split logits by attribute
        lat_logits, lon_logits, sog_logits, cog_logits = \
            torch.split(logits, (model.lat_size, model.lon_size, model.sog_size, model.cog_size), dim=-1)

        # Apply sampling modes/constraints
        if sample_mode in ("pos_vicinity",):
            # Get the last actual position indices from the *original* context or the *last generated* point
            # We need the uniform indices for vicinity sampling. Let's use the last generated point.
            # model.to_indexes is a method of EnhancTrAISformer, which should be available on `model` here.
            # model.to_indexes expects continuous values in [0,1).
            # The last point in generated_seqs is already in [0,1).
            idxs, idxs_uniform = model.to_indexes(generated_seqs[:, -1:, :])
            lat_idxs_uniform, lon_idxs_uniform = idxs_uniform[:, 0, 0:1], idxs_uniform[:, 0, 1:2] # (batch, 1)

            # Apply vicinity filtering to lat and lon logits
            lat_logits = utils.top_k_nearest_idx(lat_logits, lat_idxs_uniform, r_vicinity)
            lon_logits = utils.top_k_nearest_idx(lon_logits, lon_idxs_uniform, r_vicinity)

        if top_k is not None:
            # Apply top-k filtering to all logits
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

        # Concatenate the sampled indices for the next step
        ix = torch.cat((lat_ix, lon_ix, sog_ix, cog_ix), dim=-1) # (batch_size, 4)
        x_sample = (ix.float() + 0.5) / model.att_sizes # (batch_size, 4)

        # Append the generated step to the sequence
        generated_seqs = torch.cat((generated_seqs, x_sample.unsqueeze(1)), dim=1) # (batch_size, current_seqlen + 1, 4)

    return generated_seqs # Return the full sequence including context and generated steps



def setup_trainer(model, aisdatasets, aisdls, cf):
    # Pass the combined EnALSModel to the Trainer
    return Trainer(model, aisdatasets["train"], aisdatasets["valid"], cf, savedir=cf.savedir, device=cf.device, aisdls=aisdls, INIT_SEQLEN=cf.init_seqlen)