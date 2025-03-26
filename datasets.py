

import numpy as np
import os
import pickle

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class FourHotEncoder:
    def __init__(self, lat_range, lon_range, sog_max, n_bins, sigma_scale=0.001):
        self.sigma_scale = sigma_scale
        self.lat_bins = np.linspace(lat_range[0], lat_range[1], n_bins[0])
        self.lon_bins = np.linspace(lon_range[0], lon_range[1], n_bins[1])
        self.sog_bins = np.linspace(0, sog_max, n_bins[2])
        self.cog_bins = np.linspace(0, 360, n_bins[3])

    def gaussian_encoding(self, value, bins, sigma):
        encoding = np.exp(- (value - bins)**2 / (2 * sigma**2))
        return encoding / encoding.sum()

    def encode(self, x):
        lat_sigma = (self.lat_bins[1] - self.lat_bins[0]) * self.sigma_scale
        lon_sigma = (self.lon_bins[1] - self.lon_bins[0]) * self.sigma_scale
        sog_sigma = (self.sog_bins[1] - self.sog_bins[0]) * self.sigma_scale
        cog_sigma = (self.cog_bins[1] - self.cog_bins[0]) * self.sigma_scale

        lat_enc = self.gaussian_encoding(x[0], self.lat_bins, lat_sigma)
        lon_enc = self.gaussian_encoding(x[1], self.lon_bins, lon_sigma)
        sog_enc = self.gaussian_encoding(x[2], self.sog_bins, sog_sigma)
        cog_enc = self.gaussian_encoding(x[3], self.cog_bins, cog_sigma)

        # تحويل إلى المجال الترددي باستخدام FFT
        lat_enc_freq = np.fft.fft(lat_enc)
        lon_enc_freq = np.fft.fft(lon_enc)
        sog_enc_freq = np.fft.fft(sog_enc)
        cog_enc_freq = np.fft.fft(cog_enc)

        return np.concatenate([lat_enc_freq, lon_enc_freq, sog_enc_freq, cog_enc_freq])

    def decode(self, encoded):
        n_lat = len(self.lat_bins)
        n_lon = len(self.lon_bins)
        n_sog = len(self.sog_bins)
        n_cog = len(self.cog_bins)

        # إعادة التحويل من المجال الترددي باستخدام IFFT
        lat_enc_freq = encoded[:n_lat]
        lon_enc_freq = encoded[n_lat:n_lat+n_lon]
        sog_enc_freq = encoded[n_lat+n_lon:n_lat+n_lon+n_sog]
        cog_enc_freq = encoded[n_lat+n_lon+n_sog:]

        lat_enc = np.fft.ifft(lat_enc_freq).real
        lon_enc = np.fft.ifft(lon_enc_freq).real
        sog_enc = np.fft.ifft(sog_enc_freq).real
        cog_enc = np.fft.ifft(cog_enc_freq).real

        lat = np.sum(lat_enc * self.lat_bins)
        lon = np.sum(lon_enc * self.lon_bins)
        sog = np.sum(sog_enc * self.sog_bins)
        cog = np.sum(cog_enc * self.cog_bins)

        return np.round(np.array([lat, lon, sog, cog]), 4)

    def encode_batch(self, X):
        return np.array([self.encode(x) for x in X])

    def decode_batch(self, encoded_batch):
        return np.array([self.decode(enc) for enc in encoded_batch])


class FourHotEncoderFFT:
    def __init__(self, lat_range, lon_range, sog_max, n_bins, sigma_scale=0.001):
        self.sigma_scale = sigma_scale
        self.lat_bins = np.linspace(lat_range[0], lat_range[1], n_bins[0])
        self.lon_bins = np.linspace(lon_range[0], lon_range[1], n_bins[1])
        self.sog_bins = np.linspace(0, sog_max, n_bins[2])
        self.cog_bins = np.linspace(0, 360, n_bins[3])

    def gaussian_encoding(self, value, bins, sigma):
        encoding = np.exp(- (value - bins)**2 / (2 * sigma**2))
        return encoding / encoding.sum()

    def encode(self, x):
        lat_sigma = (self.lat_bins[1] - self.lat_bins[0]) * self.sigma_scale
        lon_sigma = (self.lon_bins[1] - self.lon_bins[0]) * self.sigma_scale
        sog_sigma = (self.sog_bins[1] - self.sog_bins[0]) * self.sigma_scale
        cog_sigma = (self.cog_bins[1] - self.cog_bins[0]) * self.sigma_scale

        lat_enc = self.gaussian_encoding(x[0], self.lat_bins, lat_sigma)
        lon_enc = self.gaussian_encoding(x[1], self.lon_bins, lon_sigma)
        sog_enc = self.gaussian_encoding(x[2], self.sog_bins, sog_sigma)
        cog_enc = self.gaussian_encoding(x[3], self.cog_bins, cog_sigma)

        # تحويل إلى المجال الترددي باستخدام FFT
        lat_enc_freq = np.fft.fft(lat_enc)
        lon_enc_freq = np.fft.fft(lon_enc)
        sog_enc_freq = np.fft.fft(sog_enc)
        cog_enc_freq = np.fft.fft(cog_enc)

        return np.concatenate([lat_enc_freq, lon_enc_freq, sog_enc_freq, cog_enc_freq])

    def decode(self, encoded):
        n_lat = len(self.lat_bins)
        n_lon = len(self.lon_bins)
        n_sog = len(self.sog_bins)
        n_cog = len(self.cog_bins)

        # إعادة التحويل من المجال الترددي باستخدام IFFT
        lat_enc_freq = encoded[:n_lat]
        lon_enc_freq = encoded[n_lat:n_lat+n_lon]
        sog_enc_freq = encoded[n_lat+n_lon:n_lat+n_lon+n_sog]
        cog_enc_freq = encoded[n_lat+n_lon+n_sog:]

        lat_enc = np.fft.ifft(lat_enc_freq).real
        lon_enc = np.fft.ifft(lon_enc_freq).real
        sog_enc = np.fft.ifft(sog_enc_freq).real
        cog_enc = np.fft.ifft(cog_enc_freq).real

        lat = np.sum(lat_enc * self.lat_bins)
        lon = np.sum(lon_enc * self.lon_bins)
        sog = np.sum(sog_enc * self.sog_bins)
        cog = np.sum(cog_enc * self.cog_bins)

        return np.round(np.array([lat, lon, sog, cog]), 4)

    def encode_batch(self, X):
        return np.array([self.encode(x) for x in X])

    def decode_batch(self, encoded_batch):
        return np.array([self.decode(enc) for enc in encoded_batch])
        
class AISDataset(Dataset):
    """Customized Pytorch dataset.
    """
    def __init__(self, 
                 l_data, 
                 max_seqlen=96,
                 dtype=torch.float32,
                 device=torch.device("cpu")):
        """
        Args
            l_data: list of dictionaries, each element is an AIS trajectory. 
                l_data[idx]["mmsi"]: vessel's MMSI.
                l_data[idx]["traj"]: a matrix whose columns are 
                    [LAT, LON, SOG, COG, TIMESTAMP]
                lat, lon, sog, and cod have been standardized, i.e. range = [0,1).
            max_seqlen: (optional) max sequence length. Default is
        """    
            
        self.max_seqlen = max_seqlen
        self.device = device
        
        self.l_data = l_data 

    def __len__(self):
        return len(self.l_data)
        
    def __getitem__(self, idx):
        """Gets items.
        
        Returns:
            seq: Tensor of (max_seqlen, [lat,lon,sog,cog]).
            mask: Tensor of (max_seqlen, 1). mask[i] = 0.0 if x[i] is a
            padding.
            seqlen: sequence length.
            mmsi: vessel's MMSI.
            time_start: timestamp of the starting time of the trajectory.
        """
        V = self.l_data[idx]
        m_v = V["traj"][:,:4] # lat, lon, sog, cog
#         m_v[m_v==1] = 0.9999
        m_v[m_v>0.9999] = 0.9999
        seqlen = min(len(m_v), self.max_seqlen)
        seq = np.zeros((self.max_seqlen,4))
        seq[:seqlen,:] = m_v[:seqlen,:]
        seq = torch.tensor(seq, dtype=torch.float32)
        
        mask = torch.zeros(self.max_seqlen)
        mask[:seqlen] = 1.
        
        seqlen = torch.tensor(seqlen, dtype=torch.int)
        mmsi =  torch.tensor(V["mmsi"], dtype=torch.int)
        time_start = torch.tensor(V["traj"][0,4], dtype=torch.int)
        #V["traj"][0, 4].clone().detach().to(torch.int)#
        return seq , mask, seqlen, mmsi, time_start
    
class AISDataset_grad(Dataset):
    """Customized Pytorch dataset.
    Return the positions and the gradient of the positions.
    """
    def __init__(self, 
                 l_data, 
                 dlat_max=0.04,
                 dlon_max=0.04,
                 max_seqlen=96,
                 dtype=torch.float32,
                 device=torch.device("cpu")):
        """
        Args
            l_data: list of dictionaries, each element is an AIS trajectory. 
                l_data[idx]["mmsi"]: vessel's MMSI.
                l_data[idx]["traj"]: a matrix whose columns are 
                    [LAT, LON, SOG, COG, TIMESTAMP]
                lat, lon, sog, and cod have been standardized, i.e. range = [0,1).
            dlat_max, dlon_max: the maximum value of the gradient of the positions.
                dlat_max = max(lat[idx+1]-lat[idx]) for all idx.
            max_seqlen: (optional) max sequence length. Default is
        """    
            
        self.dlat_max = dlat_max
        self.dlon_max = dlon_max
        self.dpos_max = np.array([dlat_max, dlon_max])
        self.max_seqlen = max_seqlen
        self.device = device
        
        self.l_data = l_data 

    def __len__(self):
        return len(self.l_data)
        
    def __getitem__(self, idx):
        """Gets items.
        
        Returns:
            seq: Tensor of (max_seqlen, [lat,lon,sog,cog]).
            mask: Tensor of (max_seqlen, 1). mask[i] = 0.0 if x[i] is a
            padding.
            seqlen: sequence length.
            mmsi: vessel's MMSI.
            time_start: timestamp of the starting time of the trajectory.
        """
        V = self.l_data[idx]
        m_v = V["traj"][:,:4] # lat, lon, sog, cog
        m_v[m_v==1] = 0.9999
        seqlen = min(len(m_v), self.max_seqlen)
        seq = np.zeros((self.max_seqlen,4))
        # lat and lon
        seq[:seqlen,:2] = m_v[:seqlen,:2] 
        # dlat and dlon
        dpos = (m_v[1:,:2]-m_v[:-1,:2]+self.dpos_max )/(2*self.dpos_max )
        dpos = np.concatenate((dpos[:1,:],dpos),axis=0)
        dpos[dpos>=1] = 0.9999
        dpos[dpos<=0] = 0.0
        seq[:seqlen,2:] = dpos[:seqlen,:2] 
        
        # convert to Tensor
        seq = torch.tensor(seq, dtype=torch.float32)
        
        mask = torch.zeros(self.max_seqlen)
        mask[:seqlen] = 1.
        
        seqlen = torch.tensor(seqlen, dtype=torch.int)
        mmsi =  torch.tensor(V["mmsi"], dtype=torch.int)
        time_start = torch.tensor(V["traj"][0,4], dtype=torch.int)
        
        return seq , mask, seqlen, mmsi, time_start
