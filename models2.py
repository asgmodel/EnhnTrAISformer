
import math
import logging
import pdb # Keep pdb import if needed for debugging

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Normal, Distribution, Categorical
from torch import Tensor
from models import EnhancTrAISformer

logger = logging.getLogger(__name__)


class ReparameterizedDiagonalGaussian(Distribution):
    def __init__(self, mu, log_sigma):
        assert mu.shape == log_sigma.shape, "Mu and log_sigma shapes must match."
        self.mu = mu
        self.sigma = log_sigma.exp()
        self.base_dist = Normal(torch.zeros_like(self.mu), torch.ones_like(self.sigma))
        super().__init__(batch_shape=self.mu.size()[:-1], event_shape=self.mu.size()[-1:])
    def sample_epsilon(self): return self.base_dist.sample()
    def rsample(self, sample_shape=torch.Size()): return self.mu + self.sigma * self.sample_epsilon()
    def log_prob(self, z): return Normal(self.mu, self.sigma).log_prob(z).sum(dim=-1)
    @property
    def mean(self): return self.mu
    @property
    def variance(self): return self.sigma.pow(2)

def kl_divergence_diag_gaussians(p: ReparameterizedDiagonalGaussian, q: ReparameterizedDiagonalGaussian) -> Tensor:
    log_var_p = p.sigma.log().mul(2); log_var_q = q.sigma.log().mul(2)
    var_p = p.variance; var_q = q.variance; mu_p = p.mean; mu_q = q.mean; k = mu_p.size(-1)
    term1 = (log_var_q - log_var_p).sum(dim=-1); term2 = ((var_p + (mu_p - mu_q).pow(2)) / var_q).sum(dim=-1)
    return 0.5 * (term1 + term2 - k)


import torch
from torch import nn
from torch.distributions import Normal, Distribution
import math
from torch import Tensor


# Define Encoder, Prior, and Decoder classes as before
class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()
        self.phi_x = nn.Sequential(nn.Linear(input_dim, latent_dim), nn.ReLU())
        self.encoder_net = nn.Linear(latent_dim * 2, 2 * latent_dim)

    def forward(self, x_enc, h):
        x_enc = self.phi_x(x_enc)
        enc = self.encoder_net(torch.cat([x_enc, h[0]], dim=-1))
        mu, log_sigma = torch.chunk(enc, 2, dim=-1)
        return ReparameterizedDiagonalGaussian(mu, log_sigma), x_enc


class Prior(nn.Module):
    def __init__(self, latent_dim):
        super(Prior, self).__init__()
        self.prior_net = nn.Linear(latent_dim, 2 * latent_dim)

    def forward(self, h):
        hidden = self.prior_net(h[0])
        mu, log_sigma = torch.chunk(hidden, 2, dim=-1)
        return ReparameterizedDiagonalGaussian(mu, log_sigma)


class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Decoder, self).__init__()
        self.phi_z = nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.ReLU())
        self.decoder_net = nn.Linear(latent_dim * 2, output_dim)

    def forward(self, z, h):
        z_enc = self.phi_z(z)
        dec_input = torch.cat([z_enc, h[0]], dim=-1)
        logits = self.decoder_net(dec_input)
        return torch.sigmoid(logits), z_enc


# Define GRUState and LSTMState classes
class GRUState(nn.Module):
    def __init__(self, latent_dim):
        super(GRUState, self).__init__()
        self.gru = nn.GRU(latent_dim * 2, latent_dim, batch_first=True)

    def forward(self, x_enc, z_enc, h):
        gru_input = torch.cat([x_enc, z_enc], dim=-1).unsqueeze(1)
        _, h_next = self.gru(gru_input, h.unsqueeze(0))
        return h_next.squeeze(0), None  # No cell state needed


class LSTMState(nn.Module):
    def __init__(self, latent_dim):
        super(LSTMState, self).__init__()
        self.lstm = nn.LSTMCell(latent_dim * 2, latent_dim)

    def forward(self, x_enc, z_enc, h, c):
        lstm_input = torch.cat([x_enc, z_enc], dim=-1)
        h_next, c_next = self.lstm(lstm_input, (h, c))
        return h_next, c_next
import torch
import torch.nn as nn
import math

class HybridStateUpdate(nn.Module):
    def __init__(self, latent_dim):
        super(HybridStateUpdate, self).__init__()
        self.gru = GRUState(latent_dim)
        self.lstm = LSTMState(latent_dim)
        self.alpha = nn.Parameter(torch.tensor(1.0))  # وزن المزيج بين GRU و LSTM

    def forward(self, x_enc, z_enc, h, c=None):
        h_gru, _ = self.gru(x_enc, z_enc, h)
        h_lstm, c_new = self.lstm(x_enc, z_enc, h, c)
        h_new = self.alpha * h_gru + (self.alpha) * h_lstm

        if c is not None:
            return h_new, c_new
        else:
            return h_new, None
import numpy as np

class FourHotEncoder:
    def __init__(self, lat_range, lon_range, sog_max, n_bins, sigma_scale=0.001):
        self.sigma_scale = sigma_scale

        self.lat_bins = torch.linspace(lat_range[0], lat_range[1], n_bins[0])
        self.lon_bins = torch.linspace(lon_range[0], lon_range[1], n_bins[1])
        self.sog_bins = torch.linspace(0, sog_max, n_bins[2])
        self.cog_bins = torch.linspace(0, 360, n_bins[3])

        self.n_bins = n_bins

    def _gaussian(self, values, bins, sigma):
        # values: [N, M]
        # bins: [B]
        values = values.unsqueeze(-1)              # [N, M, 1]
        bins = bins.view(1, 1, -1)                  # [1, 1, B]
        diff = values - bins                       # [N, M, B]
        enc = torch.exp(- (diff ** 2) / (2 * sigma ** 2))
        return enc / enc.sum(dim=-1, keepdim=True)  # [N, M, B]

    def encode_batch(self, X):
        """
        X: [N, M, 4]  ← [lat, lon, sog, cog]
        return: [N, M, total_bins]
        """
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)

        # إعداد الحاويات والسجما
        lat_sigma = (self.lat_bins[1] - self.lat_bins[0]) * self.sigma_scale
        lon_sigma = (self.lon_bins[1] - self.lon_bins[0]) * self.sigma_scale
        sog_sigma = (self.sog_bins[1] - self.sog_bins[0]) * self.sigma_scale
        cog_sigma = (self.cog_bins[1] - self.cog_bins[0]) * self.sigma_scale

        lat_enc = self._gaussian(X[..., 0], self.lat_bins.to(X.device), lat_sigma)
        lon_enc = self._gaussian(X[..., 1], self.lon_bins.to(X.device), lon_sigma)
        sog_enc = self._gaussian(X[..., 2], self.sog_bins.to(X.device), sog_sigma)
        cog_enc = self._gaussian(X[..., 3], self.cog_bins.to(X.device), cog_sigma)

        return torch.cat([lat_enc, lon_enc, sog_enc, cog_enc], dim=-1)  # [N, M, total_bins]
    def decode_batch(self, encoded):
        """
        encoded: [N, M, total_bins]
        return: [N, M, 4] ← [lat, lon, sog, cog]
        """
        n_lat, n_lon, n_sog, n_cog = self.n_bins
        idx1 = n_lat
        idx2 = idx1 + n_lon
        idx3 = idx2 + n_sog
        idx4 = idx3 + n_cog

        lat_enc = encoded[..., :idx1]                  # [N, M, n_lat]
        lon_enc = encoded[..., idx1:idx2]              # [N, M, n_lon]
        sog_enc = encoded[..., idx2:idx3]              # [N, M, n_sog]
        cog_enc = encoded[..., idx3:idx4]              # [N, M, n_cog]

        lat = (lat_enc * self.lat_bins.to(encoded.device)).sum(dim=-1)
        lon = (lon_enc * self.lon_bins.to(encoded.device)).sum(dim=-1)
        sog = (sog_enc * self.sog_bins.to(encoded.device)).sum(dim=-1)
        cog = (cog_enc * self.cog_bins.to(encoded.device)).sum(dim=-1)

        return torch.stack([lat, lon, sog, cog], dim=-1)  # [N, M, 4]



class MVRNNAnomalyQuality(nn.Module):
    def __init__(self, input_dim, latent_dim, state_type="Hybrid",emb=None,path_w=None,threshold=0.0001):
        super(MVRNNAnomalyQuality, self).__init__()
        self.emb = emb

        self.encoder = Encoder(input_dim, latent_dim)
        self.prior = Prior(latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)


        # Choose the state mechanism
        if state_type == "GRU":
            self.state_update = GRUState(latent_dim)
        elif state_type == "LSTM":
            self.state_update = LSTMState(latent_dim)
        elif state_type == "Hybrid":
            self.state_update = HybridStateUpdate(latent_dim)
        else:
            raise ValueError("state_type must be 'GRU', 'LSTM', or 'Hybrid'")

        self.state_type = state_type
        self.h_0 = torch.zeros(1, latent_dim)
        self.c_0 = torch.zeros(1, latent_dim) if state_type in ["LSTM", "Hybrid"] else None
        self.threshold =threshold

        if path_w is not None:
           w=torch.load(path_w,map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
           self.load_state_dict(w)

    def forward(self, x, beta=1.0):
        if self.emb is not None:
            print(x.shape)
            x = self.emb(x)
            print(x.shape)

        batch_size, seq_len, _ = x.size()
        h = self.h_0.expand(batch_size, -1).to(x.device)
        c = self.c_0.expand(batch_size, -1).to(x.device) if self.c_0 is not None else None
        listlogis = []

        total_loss, kl_divergence, recon_loss = 0, 0, 0

        for t in range(seq_len):
            x_t = x[:, t, :]

            prior_dist = self.prior((h, c) if self.c_0 is not None else (h, None))
            posterior_dist, x_enc = self.encoder(x_t, (h, c) if self.c_0 is not None else (h, None))

            z_t = posterior_dist.rsample()
            x_recon, z_enc = self.decoder(z_t, (h, c) if self.c_0 is not None else (h, None))

            if self.state_type == "Hybrid":
                h, c = self.state_update(x_enc, z_enc, h, c)
            elif self.state_type == "LSTM":
                h, c = self.state_update(x_enc, z_enc, h, c)
            else:
                h, _ = self.state_update(x_enc, z_enc, h)

            kl_div = kl_divergence_diag_gaussians(posterior_dist, prior_dist).sum(dim=-1)
            kl_divergence += kl_div.mean()
            recon_loss += nn.functional.binary_cross_entropy(x_recon, x_t, reduction='sum')
            listlogis.append(x_recon)

        recon_loss /= seq_len*10
        kl_divergence /= seq_len*10
        total_loss = recon_loss + beta * kl_divergence
        return total_loss, recon_loss, kl_divergence

    def get_logis(self, x):
        if self.emb is not None:
            x = self.emb(x)
        batch_size, seq_len, _ = x.size()
        h = self.h_0.expand(batch_size, -1).to(x.device)
        c = self.c_0.expand(batch_size, -1).to(x.device) if self.c_0 is not None else None
        listlogis = []

        for t in range(seq_len):
            x_t = x[:, t, :]

            prior_dist = self.prior((h, c) if self.c_0 is not None else (h, None))
            posterior_dist, x_enc = self.encoder(x_t, (h, c) if self.c_0 is not None else (h, None))
            z_t = posterior_dist.rsample()
            x_recon, z_enc = self.decoder(z_t, (h, c) if self.c_0 is not None else (h, None))
            listlogis.append(x_recon)

            if self.state_type == "Hybrid":
                h, c = self.state_update(x_enc, z_enc, h, c)
            elif self.state_type == "LSTM":
                h, c = self.state_update(x_enc, z_enc, h, c)
            else:
                h, _ = self.state_update(x_enc, z_enc, h)

        x_rect = torch.stack(listlogis, dim=0)
        x_rect = x_rect.permute(1, 0, 2)
        return x_rect

    def calculate_anomaly_rate(self, inputs):
        if self.emb is not None:
            inputs = self.emb(inputs)
        batch_size = inputs.size(0)
        h = self.h_0.expand(batch_size, -1).contiguous().to(inputs.device)
        c = self.c_0.expand(batch_size, -1).contiguous().to(inputs.device) if self.c_0 is not None else None

        total_anomalies = 0
        total_points = 0

        for t in range(inputs.size(1)):
            x = inputs[:, t, :]
            posterior_dist, x_enc = self.encoder(x, (h, c) if self.c_0 is not None else (h, None))
            z = posterior_dist.rsample()
            x_recon, z_enc = self.decoder(z, (h, c) if self.c_0 is not None else (h, None))

            diff = torch.abs(x - x_recon)
            anomalies = (diff > self.threshold).sum(dim=-1)

            total_anomalies += anomalies.sum().item()
            total_points += x.numel()

            if self.state_type == "Hybrid":
                h, c = self.state_update(x_enc, z_enc, h, c)
            elif self.state_type == "LSTM":
                h, c = self.state_update(x_enc, z_enc, h, c)
            else:
                h, _ = self.state_update(x_enc, z_enc, h)

        anomaly_rate = total_anomalies / total_points
        return anomaly_rate

    def calc_mi(self, inputs):
        if self.emb is not None:
            inputs = self.emb(inputs)
        batch_size = inputs.size(0)
        h = self.h_0.expand(batch_size, -1).contiguous().to(inputs.device)
        c = self.c_0.expand(batch_size, -1).contiguous().to(inputs.device) if self.c_0 is not None else None

        neg_entropy = 0
        log_qz = 0

        for t in range(inputs.size(1)):
            x = inputs[:, t, :]
            posterior_dist, x_enc = self.encoder(x, (h, c) if self.c_0 is not None else (h, None))
            pz = self.prior((h, c) if self.c_0 is not None else h)

            mu = posterior_dist.mu
            logsigma = torch.log(posterior_dist.sigma)
            z = posterior_dist.rsample()
            _, z_enc = self.decoder(z, (h, c) if self.c_0 is not None else (h, None))

            if self.state_type == "Hybrid":
                h, c = self.state_update(x_enc, z_enc, h, c)
            elif self.state_type == "LSTM":
                h, c = self.state_update(x_enc, z_enc, h, c)
            else:
                h, _ = self.state_update(x_enc, z_enc, h)

            neg_entropy += (-0.5 * self.encoder.encoder_net.out_features // 2 * math.log(2 * math.pi)
                            - 0.5 * (1 + 2 * logsigma).sum(-1)).mean()

            var = logsigma.exp() ** 2
            z = z.unsqueeze(1)
            mu = mu.unsqueeze(0)
            logsigma = logsigma.unsqueeze(0)
            dev = z - mu
            log_density = -0.5 * (dev ** 2 / var).sum(dim=-1) - 0.5 * (
                self.encoder.encoder_net.out_features // 2 * math.log(2 * math.pi) + (2 * logsigma).sum(dim=-1))
            log_qz1 = torch.logsumexp(log_density, dim=1) - math.log(batch_size)
            log_qz += log_qz1.mean(-1)

        mi = (neg_entropy / inputs.size(1)) - (log_qz / inputs.size(1))
        return mi


from ConfigModel import ConfigEnhancTrAISformer

class EnALSModel(nn.Module):
    """
    Enhanced Anomaly-aware Latent Space Model (EnALSModel) for AIS Trajectories.

    Integrates trajectory prediction (EnhancTrAISformer) with anomaly detection
    (MVRNNAnomalyQuality) based on a multi-stage processing workflow:

    Workflow Stages Mapping:
    1. Anomaly Detection: Handled by `self.anomaly_model` operating on input `x`.
       Provides anomaly scores/losses (e.g., ELBO). Relevant methods:
       `get_anomaly_scores`, `get_anomaly_rate`, `get_anomaly_loss_components`.
    2. Quantization ("Four Hot"): Input `x` is quantized via `self.to_indexes`
       for the Transformer input.
    3. Route Prediction: Handled by the Transformer core (`self.blocks`, `self.head`).
       Predicts the next quantized state based on the quantized input sequence.
    4. Continuous Evaluation: Primarily external. The model provides losses
       (`transformer_loss`, `anomaly_loss`, `total_loss`) as part of this stage
       when `with_targets=True`.
    5. Feedback to Anomaly Detection (Post-Prediction): *Not* part of the standard
       `forward` pass. Requires generating a trajectory and then calling anomaly
       methods on the *generated* sequence separately.
    6. Periodic Improvement: Represents the external training loop updating model
       parameters based on the combined loss.

    The model takes continuous normalized AIS data `x` as input. Internally,
    it uses `x` for the anomaly model and quantized indices derived from `x`
    for the Transformer prediction model.
    """

    def __init__(self, config, partition_model=None):
        """
        Initializes the EnALSModel.

        Args:
            config: Configuration object. Must contain attributes required by
                    EnhancTrAISformer, plus specific attributes for the anomaly module:
                    - anomaly_latent_dim (int): Latent dimension for MVRNN.
                    - anomaly_state_type (str): RNN cell type ('GRU', 'LSTM', 'Hybrid').
                    - anomaly_loss_weight (float): Weight for anomaly loss component.
                    - anomaly_encoder_hidden_dim, anomaly_decoder_hidden_dim,
                      anomaly_rnn_input_feature_dim (int, optional): MVRNN hidden dims.
                    - anomaly_threshold (float, optional): Default anomaly rate threshold.
            partition_model: Optional partition model (passed to EnhancTrAISformer).
        """
        # 1. Initialize the base Transformer model
        super(EnALSModel, self).__init__()
        self.model=EnhancTrAISformer(ConfigEnhancTrAISformer(), partition_model)
        self.config = config # Store config
        self.emb = FourHotEncoder(lat_range=[-1*config.n_lat_embd, config.n_lat_embd],
                                      lon_range=[-1*config.n_lon_embd, config.n_lon_embd],
                                      sog_max=config.n_sog_embd, n_bins=[100, 100, 50, 10], sigma_scale=0.9)


        # 2. Validate and store anomaly-specific config
        assert hasattr(config, 'anomaly_latent_dim'), "Config must include 'anomaly_latent_dim'"
        assert hasattr(config, 'anomaly_state_type'), "Config must include 'anomaly_state_type'"
        assert hasattr(config, 'anomaly_loss_weight'), "Config must include 'anomaly_loss_weight'"

        self.anomaly_loss_weight = config.anomaly_loss_weight

        # 3. Initialize the Anomaly Detection module (MVRNN)
        # Input dimension for anomaly model is the raw feature dimension (e.g., 4)
        input_dim_anomaly =260 # Lat, Lon, SOG, COG



        self.anomaly_model = MVRNNAnomalyQuality(
            input_dim=input_dim_anomaly,
            latent_dim=config.anomaly_latent_dim,
            state_type=config.anomaly_state_type,
            emb=self.emb.encode_batch,
            path_w=config.anomaly_path_w
        # encoder_hidden_dim=getattr(config, 'anomaly_encoder_hidden_dim', None),
        #     decoder_hidden_dim=getattr(config, 'anomaly_decoder_hidden_dim', None),
        #     rnn_input_feature_dim=getattr(config, 'anomaly_rnn_input_feature_dim', None),
        #     anomaly_threshold=getattr(config, 'anomaly_threshold', 0.0001)
        )

        logger.info("Initialized EnALSModel integrating EnhancTrAISformer and MVRNNAnomalyQuality.")
        logger.info(f"  Anomaly Latent Dim: {config.anomaly_latent_dim}, State: {config.anomaly_state_type}, Loss Weight: {self.anomaly_loss_weight:.4f}")
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"  Total Trainable Parameters: {total_params:e}")

    def  getsuper(self):
         return super()
    def forward(self, x, masks=None, with_targets=False, return_loss_tuple=False, anomaly_beta=1.0):
        """
        Forward pass for the combined Transformer and Anomaly Detection model.

        Args:
            x (Tensor): Input tensor (batch, seqlen, 4). Values normalized [0,1).
            masks (Tensor, optional): Mask for Transformer targets (batch, seqlen-1).
                                      1 for valid, 0 for padding.
            with_targets (bool): If True, calculate losses. Transformer uses x[:,:-1]
                                 to predict x[:,1:]. Anomaly model evaluates x[:,1:].
            return_loss_tuple (bool): If True and with_targets, return detailed losses.
            anomaly_beta (float): Beta weight for KL term in anomaly ELBO loss.

        Returns:
            Tuple containing:
            - transformer_logits: Tuple of logits from Transformer head, split by
                                  attribute (lat, lon, sog, cog). Shape e.g.,
                                  (batch, input_seqlen, lat_size).
            - total_loss (Tensor or None): Combined loss (Transf_loss + weight * Anomaly_loss).
            - loss_components (Tuple or None): If return_loss_tuple and with_targets,
                                               contains (transf_loss, anomaly_loss,
                                               anomaly_recon, anomaly_kl, transf_lat_loss, ...)
        """

        # --- Stage 3: Route Prediction (Transformer) ---
        # Request split logits and individual losses from the parent's forward method
        transformer_split_logits, transformer_loss, transformer_loss_tuple_mean =self.model.forward(
            x, masks=masks, with_targets=with_targets, return_loss_tuple=True # Get tuple internally
        )


        total_loss = None
        anomaly_loss = None
        anomaly_recon_loss = None
        anomaly_kl_div = None

        # --- Stage 1: Anomaly Detection (on input sequence) ---
        if with_targets:
            # Anomaly model evaluates the sequence corresponding to the targets
            if x.size(1) > 1:
                anomaly_input = x[:, 1:, :].contiguous()
                # Calculate ELBO loss components for the anomaly model
                anomaly_loss, anomaly_recon_loss, anomaly_kl_div = self.anomaly_model(
                    anomaly_input, beta=anomaly_beta
                )
            else: # Handle sequences too short for anomaly calc on target part
                 device = transformer_loss.device if transformer_loss is not None else x.device
                 anomaly_loss = torch.tensor(0.0, device=device)
                 anomaly_recon_loss = torch.tensor(0.0, device=device)
                 anomaly_kl_div = torch.tensor(0.0, device=device)

            # --- Combine Losses (used for Stage 4/6) ---
            if transformer_loss is not None and anomaly_loss is not None:
                total_loss = transformer_loss + self.anomaly_loss_weight * anomaly_loss

            elif transformer_loss is not None:
                 total_loss = transformer_loss # Use only transformer loss if anomaly loss couldn't be computed

        # --- Prepare return values ---
        if return_loss_tuple and with_targets:
            t_lat, t_lon, t_sog, t_cog = transformer_loss_tuple_mean if transformer_loss_tuple_mean is not None else (None,)*4
            loss_components = (
                transformer_loss, anomaly_loss, anomaly_recon_loss, anomaly_kl_div,
                t_lat, t_lon, t_sog, t_cog
            )
            return transformer_split_logits, total_loss, loss_components
        else:
            # Return only logits and combined loss (or None)
            return transformer_split_logits, total_loss

    # --- Helper Methods related to Workflow Stages ---

    def sample_next_step(self, x_context, temperature=1.0, sample=True, top_k=None):
        """
        Samples the next time step's quantized indices using the Transformer.
        Relates to Stage 3 (Prediction).

        Args:
            x_context (Tensor): Input context sequence (batch, seqlen, 4), normalized [0,1).
            temperature (float): Softmax temperature for sampling. Lower values make
                                 sampling more greedy.
            sample (bool): If True, sample from the distribution. If False, take argmax (greedy).
            top_k (int, optional): If set, restrict sampling to the top k most likely tokens.

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor]: Predicted indices for (lat, lon, sog, cog)
                                                   for the next time step. Shape: (batch, 1).
        """
        self.eval() # Ensure evaluation mode
        with torch.no_grad():
            # Use the standard forward pass to get logits for the last step in context
            split_logits, _ = self(x_context, with_targets=False)
            split_logits = split_logits[:, -1, :] / temperature
            lat_logits, lon_logits, sog_logits, cog_logits = torch.split(split_logits, (self.model.lat_size,
                                                                                        self.model.lon_size,
                                                                                        self.model.sog_size,
                                                                                        self.model.cog_size), dim=-1)

            # Get logits for the very last prediction step
            lat_logits = lat_logits / temperature
            lon_logits = lon_logits / temperature
            sog_logits = sog_logits/ temperature
            cog_logits = cog_logits/ temperature

            # Optional Top-k filtering
            if top_k is not None:
                lat_logits = self._top_k_logits(lat_logits, top_k)
                lon_logits = self._top_k_logits(lon_logits, top_k)
                sog_logits = self._top_k_logits(sog_logits, top_k)
                cog_logits = self._top_k_logits(cog_logits, top_k)

            # Get probabilities
            lat_probs = F.softmax(lat_logits, dim=-1)
            lon_probs = F.softmax(lon_logits, dim=-1)
            sog_probs = F.softmax(sog_logits, dim=-1)
            cog_probs = F.softmax(cog_logits, dim=-1)

            if sample:
                # Sample from the distribution
                lat_ix = torch.multinomial(lat_probs, num_samples=1)
                lon_ix = torch.multinomial(lon_probs, num_samples=1)
                sog_ix = torch.multinomial(sog_probs, num_samples=1)
                cog_ix = torch.multinomial(cog_probs, num_samples=1)
            else:
                # Greedy decoding (take the most likely index)
                _, lat_ix = torch.topk(lat_probs, k=1, dim=-1)
                _, lon_ix = torch.topk(lon_probs, k=1, dim=-1)
                _, sog_ix = torch.topk(sog_probs, k=1, dim=-1)
                _, cog_ix = torch.topk(cog_probs, k=1, dim=-1)

        return lat_ix, lon_ix, sog_ix, cog_ix # Shape: (batch, 1) each

    def _top_k_logits(self, logits, k):
        """Helper function for top-k sampling."""
        v, _ = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[:, [-1]]] = -float('Inf')
        return out

    def indices_to_continuous(self, indices):
        """
        Converts predicted quantized indices back to approximate continuous values [0, 1).
        Relates conceptually to Stage 2 ("Four Hot Inverse") applied to predictions.

        Args:
            indices (Tensor): Tensor of indices, shape (batch, seqlen, 4) or (batch, 4).

        Returns:
            Tensor: Tensor of approximate continuous values, same shape as input indices.
        """
        if indices.dim() == 3: # Batch, Seqlen, Features
             bs, sl, dim = indices.shape
             indices_flat = indices.view(-1, dim)
        elif indices.dim() == 2: # Batch, Features
             bs = indices.shape[0]
             sl = 1
             dim = indices.shape[1]
             indices_flat = indices
        else:
             raise ValueError("Input indices must have 2 or 3 dimensions.")

        assert dim == 4, "Indices tensor must have 4 features (lat, lon, sog, cog)"
        device = indices.device
        att_sizes = self.model.att_sizes.to(device) # [lat_size, lon_size, ...]

        # Approximate continuous value by taking the middle of the bin
        # value = (index + 0.5) / num_bins
        continuous_values = (indices_flat.float() + 0.5) / att_sizes

        # Clamp to ensure values are within [0, 1) - useful if indices might be max value
        continuous_values = torch.clamp(continuous_values, 0.0, 1.0 - 1e-6)

        if indices.dim() == 3:
            return continuous_values.view(bs, sl, dim)
        else:
            return continuous_values # Shape (batch, 4)

    # --- Anomaly-Specific Helper Methods (delegate to anomaly_model) ---

    @torch.no_grad()
    def get_anomaly_scores(self, x, score_type='recon_error', threshold=None):
        """ Calculates step-wise anomaly scores using the MVRNN (Stage 1). """
        self.anomaly_model.eval()
        # (Implementation as before, calling MVRNN's internal logic)
        # ... calculation logic ...
        batch_size, seq_len, _ = x.size(); device = x.device
        h = self.anomaly_model.h_0.expand(batch_size, -1).to(device)
        c = self.anomaly_model.c_0.expand(batch_size, -1).to(device) if self.anomaly_model._requires_cell_state else None
        scores = torch.zeros(batch_size, seq_len, device=device)
        for t in range(seq_len):
            x_t = x[:, t, :]; state_tuple = (h, c) if self.anomaly_model._requires_cell_state else h
            prior_dist = self.anomaly_model.prior(state_tuple)
            posterior_dist, x_features = self.anomaly_model.encoder(x_t, state_tuple)
            z_t = posterior_dist.rsample()
            x_recon_logits, z_features = self.anomaly_model.decoder(z_t, state_tuple)
            x_recon_probs = torch.sigmoid(x_recon_logits)
            if score_type == 'recon_error': score_t = torch.abs(x_t - x_recon_probs).mean(dim=-1)
            elif score_type == 'recon_prob': score_t = F.binary_cross_entropy(x_recon_probs, x_t, reduction='none').sum(dim=-1)
            elif score_type == 'kl_div': score_t = kl_divergence_diag_gaussians(posterior_dist, prior_dist)
            elif score_type == 'elbo': recon_term = F.binary_cross_entropy(x_recon_probs, x_t, reduction='none').sum(dim=-1); kl_term = kl_divergence_diag_gaussians(posterior_dist, prior_dist); score_t = recon_term + kl_term
            else: raise ValueError(f"Unknown score_type: {score_type}")
            scores[:, t] = score_t
            if self.anomaly_model.state_type == "GRU": h, _ = self.anomaly_model.state_update(x_features, z_features, h)
            elif self.anomaly_model.state_type == "LSTM": h, c = self.anomaly_model.state_update(x_features, z_features, h, c)
            elif self.anomaly_model.state_type == "Hybrid": h, c = self.anomaly_model.state_update(x_features, z_features, h, c)
        return scores


    @torch.no_grad()
    def get_reconstruction(self, x):
        """ Gets MVRNN reconstruction of input x (Stage 1 output / related to Stage 2 concept). """
        self.anomaly_model.eval()
        return self.anomaly_model.get_logis(x) # Assuming get_logis returns reconstructions

    @torch.no_grad()
    def get_anomaly_rate(self, x, threshold=None):
        """ Calculates overall anomaly rate via MVRNN (Stage 1 output). """
        self.anomaly_model.eval()
        return self.anomaly_model.calculate_anomaly_rate(x, threshold=threshold)

    @torch.no_grad()
    def get_mutual_information(self, x):
        """ Calculates MI I(x; z) estimate via MVRNN (Stage 1 analysis). """
        self.anomaly_model.eval()
        return self.anomaly_model.calc_mi(x)

    @torch.no_grad()
    def get_anomaly_loss_components(self, x, beta=1.0):
        """ Calculates anomaly loss components (ELBO, Recon, KL) via MVRNN (Stage 1 analysis). """
        self.anomaly_model.eval()
        return self.anomaly_model(x, beta=beta) # Call MVRNN forward
