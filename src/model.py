'''
@Description: In User Settings Edit
@Author: your name
@Date: 2019-08-25 16:21:38
@LastEditTime: 2019-09-02 14:18:29
@LastEditors: Please set LastEditors
'''
import torch
import torch.nn as nn
from allennlp.nn.util import sequence_cross_entropy_with_logits

from embeder import EmbederConfig
from embeder import Embeder
from bidcgcn import get_dcgcn
from bidcgcn import DCGCNConfig
from decoder import DecoderConfig
from decoder import Decoder
import constants as C


def build_encoder(args):
    config = DCGCNConfig(input_dim=args.emb_dim * 2,
                         output_dim=args.hid_dim,
                         num_layers=args.num_layers[0],
                         activation="relu",
                         bidirection=args.encoder_bidirection,
                         dropout=args.encoder_dropout)
    print("Dcgcn encoder config:\n", config)
    encoder = get_dcgcn(config)
    return encoder


def build_decoder(args, vocab_size):
    decoder_config = DecoderConfig(num_token=vocab_size,
                                   emb_dim=args.emb_dim,
                                   hid_dim=args.hid_dim,
                                   coverage=args.coverage,
                                   cell_type=args.decoder_cell)
    print("RNN decoder config:\n", decoder_config)
    decoder = Decoder(decoder_config)
    return decoder


def build_embeder(num_emb, emb_dim, scale_grad_by_freq, pad_id):
    embeder_config = EmbederConfig(num_emb=num_emb,
                                   emb_dim=emb_dim,
                                   padding_idx=pad_id,
                                   scale_grad_by_freq=scale_grad_by_freq)
    print("Embedder config:\n", embeder_config)
    embeder = Embeder(embeder_config)
    return embeder


def build_model(args, logger, cuda_device, vocab, edge_vocab):
    logger.info("Build node embedder...")
    node_embeder = build_embeder(num_emb=len(vocab),
                                 emb_dim=args.emb_dim,
                                 scale_grad_by_freq=args.scale_grad_by_freq,
                                 pad_id=C.PAD_ID)
    logger.info("Build pos embedder...")
    pos_embeder = build_embeder(num_emb=args.max_seq_len[0],
                                emb_dim=args.emb_dim,
                                scale_grad_by_freq=args.scale_grad_by_freq,
                                pad_id=0)

    logger.info("Build encoder...")
    encoder = build_encoder(args)

    logger.info("Build token embeder...")
    if args.weight_tying:
        logger.info("Encoder embeder is used also for decoder")
        token_embeder = node_embeder
    else:
        token_embeder = build_embeder(num_emb=len(vocab),
                                      emb_dim=args.emb_dim,
                                      scale_grad_by_freq=args.scale_grad_by_freq,
                                      pad_id=C.PAD_ID)

    logger.info("Build decoder...")
    decoder = build_decoder(args, len(vocab))
    projector = nn.Linear(args.hid_dim, len(vocab))

    model = Model(node_embeder=node_embeder,
                  pos_embeder=pos_embeder,
                  token_embeder=token_embeder,
                  encoder=encoder,
                  decoder=decoder,
                  projector=projector,
                  init_param=args.init_param,
                  weight_init_scale=args.weight_init_scale)
    if cuda_device is not None:
        model.to(cuda_device)
    return model


class Model(nn.Module):
    def __init__(self,
                 node_embeder,
                 pos_embeder,
                 token_embeder,
                 encoder,
                 decoder,
                 projector,
                 init_param,
                 weight_init_scale):
        super(Model, self).__init__()
        self.node_embeder = node_embeder
        self.pos_embeder = pos_embeder
        self.token_embeder = token_embeder
        self.encoder = encoder
        self.decoder = decoder
        self.projector = projector
        if init_param:
            self._init_param(weight_init_scale)

    def _init_param(self, weight_init_scale):
        for name, param in self.named_parameters():
            if "embeder" in name:
                continue
            elif 'weight_ih' in name:
                torch.nn.init.orthogonal_(param, gain=weight_init_scale)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param, gain=weight_init_scale)
            elif 'weight' in name:
                torch.nn.init.xavier_uniform_(param, gain=weight_init_scale)
            elif 'bias' in name:
                torch.nn.init.zeros_(param)

    def encode_graph(self, adjs, h, node_mask):
        """Encode graph with an encoder"""
        def get_gnode_value(value, mask):
            """
            value: B x N x H
            mask : B x N
            """
            batch_size, _, hid_size = value.size()
            idx = torch.sum(mask, dim=-1) - 1
            idx = idx.reshape(batch_size, 1, 1).expand(batch_size, 1, hid_size)
            return torch.gather(value, dim=1, index=idx).squeeze(1)
        value = self.encoder(adjs, h)
        state = get_gnode_value(value, node_mask)
        return value, state

    def decode_tokens(self, tokens, value, value_mask, state):
        """Decode tokens with decoder and graph embeddings"""
        embeddings = self.token_embeder(tokens[:, :-1])
        decoder_outputs, attns = self.decoder(embeddings, value, value_mask, state)
        logits = self.projector(decoder_outputs)
        return logits

    def forward(self, nlabel, npos, adjs, node_mask, tokens, token_mask):
        nembeding = self.node_embeder(nlabel)
        pembeding = self.pos_embeder(npos)
        h = torch.cat((nembeding, pembeding), dim=-1)

        value, state = self.encode_graph(adjs, h, node_mask)
        logits = self.decode_tokens(tokens, value, node_mask, state)

        targets = tokens[:, 1:].contiguous()
        weights = token_mask[:, 1:]
        loss = sequence_cross_entropy_with_logits(logits=logits, targets=targets, weights=weights)
        return loss

    def predict(self, start_tokens, nlabel, npos, adjs, node_mask, max_step):
        value, state = self.encode_graph(nlabel, npos, adjs, node_mask)
        if self.emb2hid is not None:
            state = self.emb2hid(state)
        if self.decoder.config.cell_type == 'LSTM':
            c1 = torch.zeros_like(state)

        emb = self.token_embeder(start_tokens)
        batch_size = emb.size(0)
        if self.decoder.config.coverage:
            cov_vec = torch.zeros(batch_size, value.size(1))
            cov_vec = cov_vec.to(value.device)
        else:
            cov_vec = None

        predictions = []
        for idx in range(max_step):
            if self.decoder.config.cell_type == 'GRU':
                state, cov_vec, similarity = self.decoder._step(emb, value, node_mask, cov_vec, state)
            elif self.decoder.config.cell_type == 'LSTM':
                state, c1, cov_vec, similarity = self.decoder._step(emb, value, node_mask, cov_vec, state, c1)
            logit = self.projector(state)
            t = torch.argmax(logit, dim=-1)
            predictions.append(t)
            emb = self.token_embeder(t)
        predictions = torch.stack(predictions, dim=-1)
        return predictions
