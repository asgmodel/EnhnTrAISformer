# coding=utf-8
# Copyright 2021, Duong Nguyen
#
# Licensed under the CECILL-C License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.cecill.info
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Models for TrAISformer.
    https://arxiv.org/abs/2109.03958

The code is built upon:
    https://github.com/karpathy/minGPT
"""

import math
import logging
import pdb


import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)

import torch
from torch import nn
from torch.distributions import Normal, Distribution
import math
from torch import Tensor

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(config.max_seqlen, config.max_seqlen))
                                     .view(1, 1, config.max_seqlen, config.max_seqlen))
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class EnhancTrAISformer(nn.Module):
    """Transformer for AIS trajectories."""

    def __init__(self, config, partition_model = None):
        super().__init__()

        self.lat_size = config.lat_size
        self.lon_size = config.lon_size
        self.sog_size = config.sog_size
        self.cog_size = config.cog_size
        self.full_size = config.full_size
        self.n_lat_embd = config.n_lat_embd
        self.n_lon_embd = config.n_lon_embd
        self.n_sog_embd = config.n_sog_embd
        self.n_cog_embd = config.n_cog_embd
        self.register_buffer(
            "att_sizes", 
            torch.tensor([config.lat_size, config.lon_size, config.sog_size, config.cog_size]))
        self.register_buffer(
            "emb_sizes", 
            torch.tensor([config.n_lat_embd, config.n_lon_embd, config.n_sog_embd, config.n_cog_embd]))
        
        if hasattr(config,"partition_mode"):
            self.partition_mode = config.partition_mode
        else:
            self.partition_mode = "uniform"
        self.partition_model = partition_model
        
        if hasattr(config,"blur"):
            self.blur = config.blur
            self.blur_learnable = config.blur_learnable
            self.blur_loss_w = config.blur_loss_w
            self.blur_n = config.blur_n
            if self.blur:
                self.blur_module = nn.Conv1d(1, 1, 3, padding = 1, padding_mode = 'replicate', groups=1, bias=False)
                if not self.blur_learnable:
                    for params in self.blur_module.parameters():
                        params.requires_grad = False
                        params.fill_(1/3)
            else:
                self.blur_module = None
                
        
        if hasattr(config,"lat_min"): # the ROI is provided.
            self.lat_min = config.lat_min
            self.lat_max = config.lat_max
            self.lon_min = config.lon_min
            self.lon_max = config.lon_max
            self.lat_range = config.lat_max-config.lat_min
            self.lon_range = config.lon_max-config.lon_min
            self.sog_range = 30.
            
        if hasattr(config,"mode"): # mode: "pos" or "velo".
            # "pos": predict directly the next positions.
            # "velo": predict the velocities, use them to 
            # calculate the next positions.
            self.mode = config.mode
        else:
            self.mode = "pos"
    

        # Passing from the 4-D space to a high-dimentional space
        self.lat_emb = nn.Embedding(self.lat_size, config.n_lat_embd)
        self.lon_emb = nn.Embedding(self.lon_size, config.n_lon_embd)
        self.sog_emb = nn.Embedding(self.sog_size, config.n_sog_embd)
        self.cog_emb = nn.Embedding(self.cog_size, config.n_cog_embd)
            
            
        self.pos_emb = nn.Parameter(torch.zeros(1, config.max_seqlen, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        
        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        
        
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        if self.mode in ("mlp_pos","mlp"):
            self.head = nn.Linear(config.n_embd, config.n_embd, bias=False)
        else:
            self.head = nn.Linear(config.n_embd, self.full_size, bias=False) # Classification head
            
        self.max_seqlen = config.max_seqlen
        self.apply(self._init_weights)

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def get_max_seqlen(self):
        return self.max_seqlen

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv1d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer
   
    
    def to_indexes(self, x, mode="uniform"):
        """Convert tokens to indexes.
        
        Args:
            x: a Tensor of size (batchsize, seqlen, 4). x has been truncated 
                to [0,1).
            model: currenly only supports "uniform".
        
        Returns:
            idxs: a Tensor (dtype: Long) of indexes.
        """
        bs, seqlen, data_dim = x.shape
        if mode == "uniform":
            idxs = (x*self.att_sizes).long()
            return idxs, idxs
        elif mode in ("freq", "freq_uniform"):
            
            idxs = (x*self.att_sizes).long()
            idxs_uniform = idxs.clone()
            discrete_lats, discrete_lons, lat_ids, lon_ids = self.partition_model(x[:,:,:2])
#             pdb.set_trace()
            idxs[:,:,0] = torch.round(lat_ids.reshape((bs,seqlen))).long()
            idxs[:,:,1] = torch.round(lon_ids.reshape((bs,seqlen))).long()                               
            return idxs, idxs_uniform
    
    
    def forward(self, x, masks = None, with_targets=False, return_loss_tuple=False):
        """
        Args:
            x: a Tensor of size (batchsize, seqlen, 4). x has been truncated 
                to [0,1).
            masks: a Tensor of the same size of x. masks[idx] = 0. if 
                x[idx] is a padding.
            with_targets: if True, inputs = x[:,:-1,:], targets = x[:,1:,:], 
                otherwise inputs = x.
        Returns: 
            logits, loss
        """
        
        if self.mode in ("mlp_pos","mlp",):
            idxs, idxs_uniform = x, x # use the real-values of x.
        else:            
            # Convert to indexes
            idxs, idxs_uniform = self.to_indexes(x, mode=self.partition_mode)
        
        if with_targets:
            inputs = idxs[:,:-1,:].contiguous()
            targets = idxs[:,1:,:].contiguous()
            targets_uniform = idxs_uniform[:,1:,:].contiguous()
            inputs_real = x[:,:-1,:].contiguous()
            targets_real = x[:,1:,:].contiguous()
        else:
            inputs_real = x
            inputs = idxs
            targets = None
            
        batchsize, seqlen, _ = inputs.size()
        assert seqlen <= self.max_seqlen, "Cannot forward, model block size is exhausted."

        # forward the GPT model
        lat_embeddings = self.lat_emb(inputs[:,:,0]) # (bs, seqlen, lat_size)
        lon_embeddings = self.lon_emb(inputs[:,:,1]) 
        sog_embeddings = self.sog_emb(inputs[:,:,2]) 
        cog_embeddings = self.cog_emb(inputs[:,:,3])      
        token_embeddings = torch.cat((lat_embeddings, lon_embeddings, sog_embeddings, cog_embeddings),dim=-1)
            
        position_embeddings = self.pos_emb[:, :seqlen, :] # each position maps to a (learnable) vector (1, seqlen, n_embd)
        fea = self.drop(token_embeddings + position_embeddings)
        fea = self.blocks(fea)
        fea = self.ln_f(fea) # (bs, seqlen, n_embd)
        logits = self.head(fea) # (bs, seqlen, full_size) or (bs, seqlen, n_embd)
        
        lat_logits, lon_logits, sog_logits, cog_logits =\
            torch.split(logits, (self.lat_size, self.lon_size, self.sog_size, self.cog_size), dim=-1)
        
        # Calculate the loss
        loss = None
        loss_tuple = None
        if targets is not None:

            sog_loss = F.cross_entropy(sog_logits.view(-1, self.sog_size), 
                                       targets[:,:,2].view(-1), 
                                       reduction="none").view(batchsize,seqlen)
            cog_loss = F.cross_entropy(cog_logits.view(-1, self.cog_size), 
                                       targets[:,:,3].view(-1), 
                                       reduction="none").view(batchsize,seqlen)
            lat_loss = F.cross_entropy(lat_logits.view(-1, self.lat_size), 
                                       targets[:,:,0].view(-1), 
                                       reduction="none").view(batchsize,seqlen)
            lon_loss = F.cross_entropy(lon_logits.view(-1, self.lon_size), 
                                       targets[:,:,1].view(-1), 
                                       reduction="none").view(batchsize,seqlen)                     

            if self.blur:
                lat_probs = F.softmax(lat_logits, dim=-1) 
                lon_probs = F.softmax(lon_logits, dim=-1)
                sog_probs = F.softmax(sog_logits, dim=-1)
                cog_probs = F.softmax(cog_logits, dim=-1)

                for _ in range(self.blur_n):
                    blurred_lat_probs = self.blur_module(lat_probs.reshape(-1,1,self.lat_size)).reshape(lat_probs.shape)
                    blurred_lon_probs = self.blur_module(lon_probs.reshape(-1,1,self.lon_size)).reshape(lon_probs.shape)
                    blurred_sog_probs = self.blur_module(sog_probs.reshape(-1,1,self.sog_size)).reshape(sog_probs.shape)
                    blurred_cog_probs = self.blur_module(cog_probs.reshape(-1,1,self.cog_size)).reshape(cog_probs.shape)

                    blurred_lat_loss = F.nll_loss(blurred_lat_probs.view(-1, self.lat_size),
                                                  targets[:,:,0].view(-1),
                                                  reduction="none").view(batchsize,seqlen)
                    blurred_lon_loss = F.nll_loss(blurred_lon_probs.view(-1, self.lon_size),
                                                  targets[:,:,1].view(-1),
                                                  reduction="none").view(batchsize,seqlen)
                    blurred_sog_loss = F.nll_loss(blurred_sog_probs.view(-1, self.sog_size),
                                                  targets[:,:,2].view(-1),
                                                  reduction="none").view(batchsize,seqlen)
                    blurred_cog_loss = F.nll_loss(blurred_cog_probs.view(-1, self.cog_size),
                                                  targets[:,:,3].view(-1),
                                                  reduction="none").view(batchsize,seqlen)

                    lat_loss += self.blur_loss_w*blurred_lat_loss
                    lon_loss += self.blur_loss_w*blurred_lon_loss
                    sog_loss += self.blur_loss_w*blurred_sog_loss
                    cog_loss += self.blur_loss_w*blurred_cog_loss

                    lat_probs = blurred_lat_probs
                    lon_probs = blurred_lon_probs
                    sog_probs = blurred_sog_probs
                    cog_probs = blurred_cog_probs
                    

            loss_tuple = (lat_loss, lon_loss, sog_loss, cog_loss)
            loss = sum(loss_tuple)
        
            if masks is not None:
                loss = (loss*masks).sum(dim=1)/masks.sum(dim=1)
        
            loss = loss.mean()
        
        if return_loss_tuple:
            return logits, loss, loss_tuple
        else:
            return logits, loss





import math
import logging
import pdb # Keep pdb import if needed for debugging

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Normal, Distribution, Categorical
from torch import Tensor

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

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim=None):
        super(Encoder, self).__init__(); hidden_dim = hidden_dim or latent_dim
        self.phi_x = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.encoder_net = nn.Linear(hidden_dim + latent_dim, 2 * latent_dim)
    def forward(self, x_enc_input, h):
        h_state = h[0] if isinstance(h, tuple) else h; x_features = self.phi_x(x_enc_input)
        combined_input = torch.cat([x_features, h_state], dim=-1); enc_params = self.encoder_net(combined_input)
        mu, log_sigma = torch.chunk(enc_params, 2, dim=-1); return ReparameterizedDiagonalGaussian(mu, log_sigma), x_features

class Prior(nn.Module):
    def __init__(self, latent_dim):
        super(Prior, self).__init__(); self.prior_net = nn.Linear(latent_dim, 2 * latent_dim)
    def forward(self, h):
        h_state = h[0] if isinstance(h, tuple) else h; hidden_params = self.prior_net(h_state)
        mu, log_sigma = torch.chunk(hidden_params, 2, dim=-1); return ReparameterizedDiagonalGaussian(mu, log_sigma)

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim, hidden_dim=None):
        super(Decoder, self).__init__(); hidden_dim = hidden_dim or latent_dim
        self.phi_z = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU())
        self.decoder_net = nn.Linear(hidden_dim + latent_dim, output_dim)
    def forward(self, z, h):
        h_state = h[0] if isinstance(h, tuple) else h; z_features = self.phi_z(z)
        dec_input = torch.cat([z_features, h_state], dim=-1); logits = self.decoder_net(dec_input)
        return logits, z_features

class GRUState(nn.Module):
    def __init__(self, latent_dim, input_feature_dim=None):
        super(GRUState, self).__init__(); input_feature_dim = input_feature_dim or latent_dim
        self.gru = nn.GRUCell(input_feature_dim * 2, latent_dim)
    def forward(self, x_features, z_features, h_prev):
        gru_input = torch.cat([x_features, z_features], dim=-1); h_next = self.gru(gru_input, h_prev)
        return h_next, None

class LSTMState(nn.Module):
    def __init__(self, latent_dim, input_feature_dim=None):
        super(LSTMState, self).__init__(); input_feature_dim = input_feature_dim or latent_dim
        self.lstm = nn.LSTMCell(input_feature_dim * 2, latent_dim)
    def forward(self, x_features, z_features, h_prev, c_prev):
        lstm_input = torch.cat([x_features, z_features], dim=-1); h_next, c_next = self.lstm(lstm_input, (h_prev, c_prev))
        return h_next, c_next

class HybridStateUpdate(nn.Module):
    def __init__(self, latent_dim, input_feature_dim=None):
        super(HybridStateUpdate, self).__init__(); input_feature_dim = input_feature_dim or latent_dim
        self.gru_state = GRUState(latent_dim, input_feature_dim); self.lstm_state = LSTMState(latent_dim, input_feature_dim)
        self.alpha_logit = nn.Parameter(torch.tensor(0.0))
    def forward(self, x_features, z_features, h_prev, c_prev):
        alpha = torch.sigmoid(self.alpha_logit); h_gru, _ = self.gru_state(x_features, z_features, h_prev)
        h_lstm, c_next = self.lstm_state(x_features, z_features, h_prev, c_prev)
        h_next = alpha * h_gru + (1.0 - alpha) * h_lstm; return h_next, c_next

class MVRNNAnomalyQuality(nn.Module):
    def __init__(self, input_dim, latent_dim, state_type="Hybrid", encoder_hidden_dim=None, decoder_hidden_dim=None, rnn_input_feature_dim=None, anomaly_threshold=0.0001):
        super(MVRNNAnomalyQuality, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim, hidden_dim=encoder_hidden_dim)
        self.prior = Prior(latent_dim); self.decoder = Decoder(latent_dim, input_dim, hidden_dim=decoder_hidden_dim)
        self.state_type = state_type; rnn_feature_dim = rnn_input_feature_dim or (encoder_hidden_dim or latent_dim)
        if state_type == "GRU": self.state_update = GRUState(latent_dim, input_feature_dim=rnn_feature_dim); self._requires_cell_state = False
        elif state_type == "LSTM": self.state_update = LSTMState(latent_dim, input_feature_dim=rnn_feature_dim); self._requires_cell_state = True
        elif state_type == "Hybrid": self.state_update = HybridStateUpdate(latent_dim, input_feature_dim=rnn_feature_dim); self._requires_cell_state = True
        else: raise ValueError("state_type must be 'GRU', 'LSTM', or 'Hybrid'")
        self.register_buffer('h_0', torch.zeros(1, latent_dim))
        if self._requires_cell_state: self.register_buffer('c_0', torch.zeros(1, latent_dim))
        else: self.c_0 = None
        self.latent_dim = latent_dim; self.input_dim = input_dim; self.threshold = anomaly_threshold
    def forward(self, x, beta=1.0):
        batch_size, seq_len, _ = x.size(); device = x.device; h = self.h_0.expand(batch_size, -1).to(device)
        c = self.c_0.expand(batch_size, -1).to(device) if self._requires_cell_state else None
        total_kl_div = 0.0; total_recon_loss = 0.0
        for t in range(seq_len):
            x_t = x[:, t, :]; state_tuple_for_prior = (h, c) if self._requires_cell_state else h
            prior_dist = self.prior(state_tuple_for_prior); state_tuple_for_encoder = (h, c) if self._requires_cell_state else h
            posterior_dist, x_features = self.encoder(x_t, state_tuple_for_encoder); z_t = posterior_dist.rsample()
            state_tuple_for_decoder = (h, c) if self._requires_cell_state else h
            x_recon_logits, z_features = self.decoder(z_t, state_tuple_for_decoder); x_recon_probs = torch.sigmoid(x_recon_logits)
            recon_loss_t = F.binary_cross_entropy(x_recon_probs, x_t, reduction='sum'); total_recon_loss += recon_loss_t
            kl_div_t = kl_divergence_diag_gaussians(posterior_dist, prior_dist).sum(); total_kl_div += kl_div_t
            state_tuple_prev = (h, c) if self._requires_cell_state else h
            if self.state_type == "GRU": h, _ = self.state_update(x_features, z_features, h)
            elif self.state_type == "LSTM": h, c = self.state_update(x_features, z_features, h, c)
            elif self.state_type == "Hybrid": h, c = self.state_update(x_features, z_features, h, c)
        avg_recon_loss = total_recon_loss / (batch_size * seq_len * self.input_dim); avg_kl_div = total_kl_div / (batch_size * seq_len)
        total_loss = avg_recon_loss + beta * avg_kl_div; return total_loss, avg_recon_loss, avg_kl_div
    def get_logis(self, x): # Renamed from original code, seems to return reconstructions (probs)
        self.eval(); batch_size, seq_len, _ = x.size(); device = x.device; h = self.h_0.expand(batch_size, -1).to(device)
        c = self.c_0.expand(batch_size, -1).to(device) if self._requires_cell_state else None; list_recons = []
        with torch.no_grad():
            for t in range(seq_len):
                x_t = x[:, t, :]; state_tuple = (h, c) if self._requires_cell_state else h
                posterior_dist, x_features = self.encoder(x_t, state_tuple); z_t = posterior_dist.rsample()
                x_recon_logits, z_features = self.decoder(z_t, state_tuple); x_recon_probs = torch.sigmoid(x_recon_logits)
                list_recons.append(x_recon_probs)
                if self.state_type == "GRU": h, _ = self.state_update(x_features, z_features, h)
                elif self.state_type == "LSTM": h, c = self.state_update(x_features, z_features, h, c)
                elif self.state_type == "Hybrid": h, c = self.state_update(x_features, z_features, h, c)
        return torch.stack(list_recons, dim=1)
    def calculate_anomaly_rate(self, inputs, threshold=None):
        self.eval(); active_threshold = threshold if threshold is not None else self.threshold
        batch_size, seq_len, input_dim = inputs.size(); device = inputs.device
        h = self.h_0.expand(batch_size, -1).to(device); c = self.c_0.expand(batch_size, -1).to(device) if self._requires_cell_state else None
        total_anomalies = 0; total_points = batch_size * seq_len * input_dim
        with torch.no_grad():
            for t in range(seq_len):
                x_t = inputs[:, t, :]; state_tuple = (h, c) if self._requires_cell_state else h
                posterior_dist, x_features = self.encoder(x_t, state_tuple); z_t = posterior_dist.rsample()
                x_recon_logits, z_features = self.decoder(z_t, state_tuple); x_recon_probs = torch.sigmoid(x_recon_logits)
                diff = torch.abs(x_t - x_recon_probs); anomalies_t = (diff > active_threshold).sum(); total_anomalies += anomalies_t.item()
                if self.state_type == "GRU": h, _ = self.state_update(x_features, z_features, h)
                elif self.state_type == "LSTM": h, c = self.state_update(x_features, z_features, h, c)
                elif self.state_type == "Hybrid": h, c = self.state_update(x_features, z_features, h, c)
        return total_anomalies / total_points if total_points > 0 else 0.0
    def calc_mi(self, inputs):
        self.eval(); batch_size, seq_len, _ = inputs.size(); device = inputs.device
        h = self.h_0.expand(batch_size, -1).to(device); c = self.c_0.expand(batch_size, -1).to(device) if self._requires_cell_state else None
        total_neg_entropy = 0.0; total_log_qz = 0.0; latent_dim = self.latent_dim
        with torch.no_grad():
            for t in range(seq_len):
                x_t = inputs[:, t, :]; state_tuple = (h, c) if self._requires_cell_state else h
                posterior_dist, x_features = self.encoder(x_t, state_tuple); mu = posterior_dist.mu; log_sigma = posterior_dist.sigma.log(); var = log_sigma.exp().pow(2)
                z_t = posterior_dist.rsample()
                log_q_z_given_x = -0.5 * (latent_dim * math.log(2 * math.pi) + (1 + 2 * log_sigma).sum(dim=-1)); total_neg_entropy += log_q_z_given_x.mean()
                z_t_expanded = z_t.unsqueeze(1); mu_expanded = mu.unsqueeze(0); log_sigma_expanded = log_sigma.unsqueeze(0); var_expanded = var.unsqueeze(0)
                dev = z_t_expanded - mu_expanded
                log_density = -0.5 * (dev.pow(2) / var_expanded).sum(dim=-1) - 0.5 * (latent_dim * math.log(2 * math.pi) + (2 * log_sigma_expanded).sum(dim=-1))
                log_qz_t = torch.logsumexp(log_density, dim=1) - math.log(batch_size); total_log_qz += log_qz_t.mean()
                _, z_features = self.decoder(z_t, state_tuple)
                if self.state_type == "GRU": h, _ = self.state_update(x_features, z_features, h)
                elif self.state_type == "LSTM": h, c = self.state_update(x_features, z_features, h, c)
                elif self.state_type == "Hybrid": h, c = self.state_update(x_features, z_features, h, c)
        avg_neg_entropy = total_neg_entropy / seq_len; avg_log_qz = total_log_qz / seq_len
        return avg_neg_entropy - avg_log_qz


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

        # 2. Validate and store anomaly-specific config
        assert hasattr(config, 'anomaly_latent_dim'), "Config must include 'anomaly_latent_dim'"
        assert hasattr(config, 'anomaly_state_type'), "Config must include 'anomaly_state_type'"
        assert hasattr(config, 'anomaly_loss_weight'), "Config must include 'anomaly_loss_weight'"

        self.anomaly_loss_weight = config.anomaly_loss_weight

        # 3. Initialize the Anomaly Detection module (MVRNN)
        # Input dimension for anomaly model is the raw feature dimension (e.g., 4)
        input_dim_anomaly = 4 # Lat, Lon, SOG, COG



        self.anomaly_model = MVRNNAnomalyQuality(
            input_dim=input_dim_anomaly,
            latent_dim=config.anomaly_latent_dim,
            state_type=config.anomaly_state_type,
            encoder_hidden_dim=getattr(config, 'anomaly_encoder_hidden_dim', None),
            decoder_hidden_dim=getattr(config, 'anomaly_decoder_hidden_dim', None),
            rnn_input_feature_dim=getattr(config, 'anomaly_rnn_input_feature_dim', None),
            anomaly_threshold=getattr(config, 'anomaly_threshold', 0.0001)
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

        
