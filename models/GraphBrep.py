import torch
import torch.nn as nn
from torch_geometric.utils import dense_to_sparse
from .hmpb import LTIABlock
from network import *


class CGTD_EP(nn.Module):
    def __init__(self, use_cf, max_p=True):
        super().__init__()
        self.embed_dim = 768
        self.use_cf = use_cf
        layer = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=12, norm_first=True, dim_feedforward=1024, dropout=0.1)
        self.net = nn.TransformerEncoder(layer, 12, nn.LayerNorm(self.embed_dim))
        self.pool = nn.AdaptiveMaxPool1d(1) if max_p else nn.AdaptiveAvgPool1d(1)
        self.num_gnn_layers = 6
        dropout = 0.1
        gnn_local = "GINE"
        gnn_global = "FullTrans_1"
        cat_dim = self.embed_dim // self.num_gnn_layers
        gnns = []
        for _ in range(self.num_gnn_layers):
            gnns.append(LTIABlock(self.embed_dim, gnn_local, gnn_global, 12, temb_dim=self.embed_dim, act=nn.SiLU(), dropout=dropout, attn_dropout=dropout))
            gnns.append(nn.Linear(self.embed_dim, cat_dim))
            gnns.append(nn.Linear(self.embed_dim, cat_dim))
        self.gnns = nn.ModuleList(gnns)

        self.surfz_embed = nn.Sequential(
            nn.Linear(3 * 16, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.SiLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

        self.surfp_embed = nn.Sequential(
            nn.Linear(6, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.SiLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

        self.edgep_embed = nn.Sequential(
            nn.Linear(6, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.SiLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

        self.time_embed = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.SiLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

        self.fc_out2 = nn.Sequential(
            nn.Linear(self.embed_dim + cat_dim * self.num_gnn_layers * 2, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.SiLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

        self.fc_pool = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.SiLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

        self.fc_out = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.SiLU(),
            nn.Linear(self.embed_dim, 6),
        )

    def forward(self, edgePos, surfPos, surfZ, timesteps, face_mask, edge_mask):
        bsz, n, _ = surfZ.shape
        time_embeds = self.time_embed(sincos_embedding(timesteps, self.embed_dim)).unsqueeze(1)
        surf_p_embeds = self.surfp_embed(surfPos)
        surf_z_embeds = self.surfz_embed(surfZ)
        surf_embeds = surf_p_embeds + surf_z_embeds
        surf_embeds_e = (surf_embeds.unsqueeze(2).repeat(1, 1, n, 1) + surf_embeds.unsqueeze(1).repeat(1, n, 1, 1)) / 2
        edge_embeds = self.edgep_embed(edgePos) * edge_mask
        data_embeds = (surf_embeds_e + edge_embeds) * edge_mask
        surf_embeds = surf_embeds + time_embeds
        tokens = (data_embeds + time_embeds.unsqueeze(1)) * edge_mask

        pooled = self.pool(edge_embeds.transpose(2, 3).flatten(0, 1)).unflatten(0, torch.Size([bsz, n])).squeeze(-1)
        global_surf = self.net(src=(surf_embeds + pooled).permute(1, 0, 2), src_key_padding_mask=~face_mask.bool()).transpose(0, 1)

        dense_ez = tokens * edge_mask
        with torch.no_grad():
            adj = edge_mask.squeeze().detach()
        h_face = surf_embeds.reshape(-1, self.embed_dim)
        dense_index = adj.nonzero(as_tuple=True)
        edge_index, _ = dense_to_sparse(adj)
        h_dense_edge = dense_ez
        face_hids, edge_hids = [], []
        m_idx = 0
        for _ in range(self.num_gnn_layers):
            h_face, h_dense_edge = self.gnns[m_idx](h_face, edge_index, h_dense_edge, dense_index, face_mask, edge_mask)
            m_idx += 1
            face_hids.append(self.gnns[m_idx](h_face.reshape(surf_embeds.shape)))
            m_idx += 1
            edge_hids.append(self.gnns[m_idx](h_dense_edge))
            m_idx += 1
        face_hids = torch.cat(face_hids, dim=-1)
        local_surf = face_hids.unsqueeze(2).repeat(1, 1, n, 1) + face_hids.unsqueeze(1).repeat(1, n, 1, 1)
        edge_hids = torch.cat(edge_hids, dim=-1)
        ep_score = self.fc_out2(torch.cat([dense_ez, edge_hids, local_surf], dim=-1) * edge_mask)
        global_edge = (global_surf.unsqueeze(2).repeat(1, 1, n, 1) + global_surf.unsqueeze(1).repeat(1, n, 1, 1)) * edge_mask
        ep_score = self.fc_pool(ep_score + global_edge)
        pred = self.fc_out(ep_score) * edge_mask
        pred = (pred + pred.transpose(1, 2)) / 2
        return pred


class CGTD_EZ(nn.Module):
    def __init__(self, use_cf, max_p=True):
        super().__init__()
        self.embed_dim = 768
        self.use_cf = use_cf
        layer = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=12, norm_first=True, dim_feedforward=1024, dropout=0.1)
        self.net = nn.TransformerEncoder(layer, 6, nn.LayerNorm(self.embed_dim))
        self.pool = nn.AdaptiveMaxPool1d(1) if max_p else nn.AdaptiveAvgPool1d(1)
        self.num_gnn_layers = 6
        dropout = 0.1
        gnn_local = "GINE"
        gnn_global = "FullTrans_1"
        cat_dim = self.embed_dim // self.num_gnn_layers
        gnns = []
        for _ in range(self.num_gnn_layers):
            gnns.append(LTIABlock(self.embed_dim, gnn_local, gnn_global, 12, temb_dim=self.embed_dim, act=nn.SiLU(), dropout=dropout, attn_dropout=dropout))
            gnns.append(nn.Linear(self.embed_dim, cat_dim))
            gnns.append(nn.Linear(self.embed_dim, cat_dim))
        self.gnns = nn.ModuleList(gnns)

        self.surfz_embed = nn.Sequential(
            nn.Linear(3 * 16, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.SiLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

        self.edgez_embed = nn.Sequential(
            nn.Linear(3 * 4, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.SiLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

        self.surfp_embed = nn.Sequential(
            nn.Linear(6, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.SiLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

        self.edgep_embed = nn.Sequential(
            nn.Linear(6, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.SiLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

        self.vertp_fc = nn.Sequential(
            nn.Linear(6, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.SiLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

        self.time_embed = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.SiLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

        self.fc_out2 = nn.Sequential(
            nn.Linear(self.embed_dim + cat_dim * self.num_gnn_layers * 2, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.SiLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

        self.fc_pool = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.SiLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

        self.fc_out = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.SiLU(),
            nn.Linear(self.embed_dim, 18),
        )

    def forward(self, edgeZ, edgePos, surfPos, surfZ, timesteps, face_mask, edge_mask):
        bsz, n, _ = surfZ.shape
        edgeZ, vertPos = edgeZ[:, :, :, :12], edgeZ[:, :, :, 12:]
        time_embeds = self.time_embed(sincos_embedding(timesteps, self.embed_dim)).unsqueeze(1)
        surf_p_embeds = self.surfp_embed(surfPos)
        surf_z_embeds = self.surfz_embed(surfZ)
        surf_embeds = surf_p_embeds + surf_z_embeds
        surf_embeds_e = (surf_embeds.unsqueeze(2).repeat(1, 1, n, 1) + surf_embeds.unsqueeze(1).repeat(1, n, 1, 1)) / 2

        edge_p_embeds = self.edgep_embed(edgePos)
        edge_z_embeds = self.edgez_embed(edgeZ)
        vert_pos_embeds = self.vertp_fc(vertPos)
        edge_embeds = edge_p_embeds + edge_z_embeds + vert_pos_embeds
        data_embeds = (surf_embeds_e + edge_embeds) * edge_mask
        surf_embeds = surf_embeds + time_embeds
        tokens = (data_embeds + time_embeds.unsqueeze(1)) * edge_mask

        pooled = self.pool(edge_embeds.transpose(2, 3).flatten(0, 1)).unflatten(0, torch.Size([bsz, n])).squeeze(-1)
        global_surf = self.net(src=(surf_embeds + pooled).permute(1, 0, 2), src_key_padding_mask=~face_mask.bool()).transpose(0, 1)

        dense_ez = tokens * edge_mask
        with torch.no_grad():
            adj = edge_mask.squeeze().detach()
        h_face = surf_embeds.reshape(-1, self.embed_dim)
        dense_index = adj.nonzero(as_tuple=True)
        edge_index, _ = dense_to_sparse(adj)
        h_dense_edge = dense_ez
        face_hids, edge_hids = [], []
        m_idx = 0
        for _ in range(self.num_gnn_layers):
            h_face, h_dense_edge = self.gnns[m_idx](h_face, edge_index, h_dense_edge, dense_index, face_mask, edge_mask)
            m_idx += 1
            face_hids.append(self.gnns[m_idx](h_face.reshape(surf_embeds.shape)))
            m_idx += 1
            edge_hids.append(self.gnns[m_idx](h_dense_edge))
            m_idx += 1
        face_hids = torch.cat(face_hids, dim=-1)
        local_surf = face_hids.unsqueeze(2).repeat(1, 1, n, 1) + face_hids.unsqueeze(1).repeat(1, n, 1, 1)
        edge_hids = torch.cat(edge_hids, dim=-1)
        ep_score = self.fc_out2(torch.cat([dense_ez, edge_hids, local_surf], dim=-1) * edge_mask)
        global_edge = (global_surf.unsqueeze(2).repeat(1, 1, n, 1) + global_surf.unsqueeze(1).repeat(1, n, 1, 1)) * edge_mask
        ep_score = self.fc_pool(ep_score + global_edge)
        pred = self.fc_out(ep_score) * edge_mask
        pred = (pred + pred.transpose(1, 2)) / 2
        return pred
