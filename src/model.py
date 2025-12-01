import torch
import torch.nn as nn
import torch.nn.functional as F




class MSAToxNet(torch.nn.Module):
    def __init__(self,embed_dim=128):
        super(MSAToxNet, self).__init__()
        self.atom_embedding = nn.Embedding(64, embed_dim)
        self.sub_embedding = nn.Embedding(498, embed_dim)
        
        # SMILES CNN
        # Atom Conv
        self.atom_conv = nn.Sequential(
            nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=3),
            nn.Softplus(),
            nn.LayerNorm(80-2),
            nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim , kernel_size=3),
        )

        # Sub-structure Conv
        self.sub_conv = nn.Sequential(
            nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=3),
            nn.Softplus(),
            nn.LayerNorm(40-2),
            nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=3),
        )
        self.fp_encoder = nn.Sequential(
            nn.Linear(2048+167, 512),
            nn.SiLU(),
            nn.LayerNorm(512),
            nn.Linear(512, 128),
        )

        self.max_pool=nn.AdaptiveMaxPool1d(1)

        # Decoder
        self.decoder=nn.Sequential(
            nn.Linear(128*2, 128),
            nn.BatchNorm1d(128),
            nn.Linear(128, 1),
        )


    def forward(self, fp, seq1, seq2):
        # Fingerprint embedding
        fp_emb=self.fp_encoder(fp)

        # SMILES embedding
        a_emb = self.atom_embedding(seq1)
        a_emb = a_emb.transpose(2, 1)
        a_emb = self.atom_conv(a_emb)
        a_emb=self.max_pool(a_emb).squeeze()

        s_emb=self.sub_embedding(seq2)
        s_emb = s_emb.transpose(2, 1)
        s_emb = self.sub_conv(s_emb)
        s_emb = self.max_pool(s_emb).squeeze()
        
        # Fusion
        rep = torch.cat((fp_emb,a_emb+s_emb), dim=1)
        out=self.decoder(rep)
        return out
