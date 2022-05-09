import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class M1(nn.Module):
    def __init__(self, cfg_params, args):
        super(M1, self).__init__()
        self.__dict__.update(args.__dict__)
        cfg_params.copyAttrib(self)
        self.gpu = self.use_cuda

        self.emb_poi = nn.Embedding(self.poi_num, self.embedding_dim)
        self.emb_loc = nn.Linear(2, self.hidden_dim)
        self.emb_user = nn.Embedding(self.user_num, self.user_embed)
        self.emb_semantic = nn.Embedding(self.category_num, int((self.hidden_dim - self.embedding_dim) / 2))
        self.emb_tid = nn.Embedding(self.tid_num, int((self.hidden_dim - self.embedding_dim) / 2))
        self.transformer = TransformerBlock(hour_len=self.obs_len, hidden_dim=self.hidden_dim)
        self.transformer_loc = TransformerBlock(hour_len=self.obs_len, hidden_dim=self.hidden_dim)
        self.transformer_social = TransformerSocial(hidden_dim=self.hidden_dim)
        self.fc = nn.Linear(self.hidden_dim + self.user_embed + self.hidden_dim, self.hidden_dim)
        self.fc_score = nn.Linear(self.hidden_dim, self.poi_num)
        self.fc_loc = nn.Linear(self.hidden_dim, 2)
        self.drop = nn.Dropout(self.drop_out)

    def forward(self, input_x, input_user, input_semantic, input_tid, input_loc, input_social, social_tid, social_semantic):

        user_emb = self.emb_user(input_user)
        batch_size = input_user.size(0)
        loc_emb = self.emb_loc(input_loc.view(-1, self.obs_len, 2))
        input_x = self.emb_poi(input_x)
        semantic_emb = self.emb_semantic(input_semantic)
        tid_emb = self.emb_tid(input_tid)
        tra_in = torch.cat((input_x, semantic_emb, tid_emb), -1)
        output = self.transformer(tra_in)
        output_loc = self.transformer_loc(loc_emb)

        if input_social.sum() != 0:
            social_seq, neighbours, social_tid_seq, social_semantic_seq = self.get_neighbour(input_social, batch_size, social_tid, social_semantic)
            social_emb = self.emb_poi(social_seq)
            social_tid_emb = self.emb_tid(social_tid_seq)
            social_semantic_emb = self.emb_semantic(social_semantic_seq)
            social_tra_in = torch.cat((social_emb, social_semantic_emb, social_tid_emb), -1)
            out_social = self.transformer(social_tra_in)
            social_relation_out = self.get_social_relation(out_social.view(-1, self.hidden_dim), output, neighbours)
            merge_features = torch.cat((social_relation_out, user_emb.squeeze(1), output_loc), -1)
        else:
            merge_features = torch.cat((output, user_emb.squeeze(1), output_loc), -1)
        pred_feature = self.drop(self.fc(merge_features))
        score = F.selu(self.fc_score(pred_feature))
        score = F.log_softmax(score, dim=1)
        pred_loc = self.fc_loc(pred_feature)
        return score, pred_loc

    def init_hidden(self, batch=1):
        if self.gpu:
            return torch.zeros(1, batch, self.hidden_dim).cuda()
        else:
            return torch.zeros(1, batch, self.hidden_dim)

    def get_social_relation(self, social_hidden, user_hidden, neighbours):
        out_total = []
        count = 0
        for i in range(len(neighbours)):
            num_neighbour = neighbours[i]
            if num_neighbour > 0:
                social = social_hidden[count: count + num_neighbour, :]
                social_transformer_in = torch.cat((social.view(num_neighbour, self.hidden_dim), user_hidden[i].view(1, self.hidden_dim)), dim=0)
                social_transformer_out = self.transformer_social(social_transformer_in.unsqueeze(0))
                out_total.append(social_transformer_out)
            else:
                out_total.append(user_hidden[i])
            count += num_neighbour

        return torch.stack(out_total, dim=0)

    def get_neighbour(self, input_social, batch_size, social_tid, social_semantic):
        neighbours = []
        out_social = input_social.view(-1, self.obs_len)
        out_tid = social_tid.view(-1, self.obs_len)
        out_semantic = social_semantic.view(-1, self.obs_len)
        out_social = out_social[out_social.sum(dim=-1) != 0]
        out_tid = out_tid[out_tid.sum(dim=-1) != 0]
        out_semantic = out_semantic[out_semantic.sum(dim=-1) != 0]
        for i in range(batch_size):
            a = torch.nonzero(input_social[i], as_tuple=True)[0]
            neighbours.append(len(torch.unique(a)))

        return out_social, neighbours, out_tid, out_semantic


class TransformerSocial(nn.Module):
    def __init__(self, hidden_dim=256, nheads=2, dropout=0.2,
                 num_encoder_layers=2):
        super(TransformerSocial, self).__init__()

        encoder_layers = nn.TransformerEncoderLayer(hidden_dim, nheads, 2*hidden_dim, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)

    def forward(self, x):
        """

        :param x:  Batch_size x len X hidden_dim
        :return: Batch_size x hidden_dim
        """
        h = x.permute(1, 0, 2)
        # sequence to the encoder
        transformer_out = self.transformer_encoder(h)

        return transformer_out[-1, :, :].squeeze(0)


class TransformerBlock(nn.Module):
    def __init__(self, hour_len, hidden_dim=256, nheads=8, dropout=0.2,
                 num_encoder_layers=6, num_decoder_layers=6):
        super(TransformerBlock, self).__init__()

        self.pos_encoder = PositionalEncoding(hidden_dim, dropout, max_len=hour_len)
        encoder_layers = nn.TransformerEncoderLayer(hidden_dim, nheads, 2*hidden_dim, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)

    def forward(self, x):
        """

        :param x:  Batch_size x len X hidden_dim
        :return: Batch_size x hidden_dim
        """
        h = x.permute(1, 0, 2)
        # sequence to the encoder
        transformer_input = self.pos_encoder(h)
        transformer_out = self.transformer_encoder(transformer_input)

        return transformer_out[-1, :, :].squeeze(0)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


