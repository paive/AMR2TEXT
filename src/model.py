import torch
import torch.nn as nn
from allennlp.nn.util import sequence_cross_entropy_with_logits

from embeder import EmbederConfig
from embeder import Embeder
from embeder import PosEmbederConfig
from embeder import PosEmbeder
from transformergcn import TransformerGCNConfig
from transformergcn import get_transfomergcn
from decoder import DecoderConfig
from decoder import Decoder
from rnn_encoder import RNNEncoder
from ensemble_encoder import EnsembleEncoder
import constants as C
from utils import deprecated


def build_gcn_encoder(args):
    config = TransformerGCNConfig(
        hid_dim=args.hid_dim,
        num_layers=args.gcn_layers,
        num_heads=args.heads,
        directions=4,
        activation="relu",
        dropout=args.model_dropout[0],
        stadia=args.stadia,
        bigcn=args.bigcn)
    encoder = get_transfomergcn(config)
    return encoder


def build_rnn_encoder(args):
    return RNNEncoder(args.hid_dim, args.rnn_layers, args.model_dropout[0])


def build_ensemble_encoder(args):
    gcn_encoder = build_gcn_encoder(args)
    rnn_encoder = build_rnn_encoder(args)
    ensemble_encoder = EnsembleEncoder(gcn_encoder, rnn_encoder, args.hid_dim)
    return ensemble_encoder


def build_encoder(args):
    if args.encoder_type == 'gcn':
        encoder = build_gcn_encoder(args)
    elif args.encoder_type == 'rnn':
        encoder = build_rnn_encoder(args)
    elif args.encoder_type == 'both':
        encoder = build_ensemble_encoder(args)
    else:
        raise NotImplementedError
    return encoder


def build_decoder(args, vocab_size):
    decoder_config = DecoderConfig(num_token=vocab_size,
                                   emb_dim=args.emb_dim,
                                   hid_dim=args.hid_dim,
                                   coverage=args.coverage,
                                   cell_type=args.decoder_cell,
                                   dropout=args.model_dropout[1])
    decoder = Decoder(decoder_config)
    return decoder


def build_embeder(num_emb, emb_dim, scale_grad_by_freq, pad_id, dropout):
    embeder_config = EmbederConfig(num_emb=num_emb,
                                   emb_dim=emb_dim,
                                   padding_idx=pad_id,
                                   scale_grad_by_freq=scale_grad_by_freq,
                                   dropout=dropout)
    embeder = Embeder(embeder_config)
    return embeder


def build_posembeder(max_seq_len, emb_dim, pad_idx):
    posembeder_config = PosEmbederConfig(max_seq_len=max_seq_len,
                                         emb_dim=emb_dim,
                                         padding_idx=pad_idx)
    posembeder = PosEmbeder(posembeder_config)
    return posembeder


def build_model(args, logger, cuda_device, vocab, edge_vocab):
    node_embeder = build_embeder(num_emb=len(vocab),
                                 emb_dim=args.emb_dim,
                                 scale_grad_by_freq=args.scale_grad_by_freq,
                                 pad_id=C.PAD_ID,
                                 dropout=args.emb_dropout[0])
    logger.info(f"Build encoder {args.encoder_type}...")

    if args.encoder_type == 'gcn':
        emb2hid = nn.Linear(args.emb_dim + args.pos_emb_dim, args.hid_dim)
        graph_emb2hid = None
        linear_emb2hid = None
        pos_embeder = build_posembeder(
            max_seq_len=args.max_seq_len[0],
            emb_dim=args.pos_emb_dim,
            pad_idx=0)
    elif args.encoder_type == 'rnn':
        emb2hid = nn.Linear(args.emb_dim, args.hid_dim)
        graph_emb2hid = None
        linear_emb2hid = None
        pos_embeder = None
    elif args.encoder_type == 'both':
        emb2hid = None
        graph_emb2hid = nn.Linear(args.emb_dim + args.pos_emb_dim, args.hid_dim)
        linear_emb2hid = nn.Linear(args.emb_dim, args.hid_dim)
        pos_embeder = build_posembeder(
            max_seq_len=args.max_seq_len[0],
            emb_dim=args.pos_emb_dim,
            pad_idx=0)

    logger.info("Build encoder...")
    encoder = build_encoder(args)

    logger.info("Build token embeder...")
    if args.weight_tying:
        logger.info("Encoder embeder is used also for decoder")
        token_embeder = node_embeder
    else:
        token_embeder = build_embeder(num_emb=len(vocab),
                                      emb_dim=args.emb_dim,
                                      hid_dim=args.hid_dim,
                                      scale_grad_by_freq=args.scale_grad_by_freq,
                                      pad_id=C.PAD_ID,
                                      dropout=args.emb_dropout[1])

    logger.info("Build decoder...")
    decoder = build_decoder(args, len(vocab))
    projector = nn.Linear(args.hid_dim, len(vocab))

    model = Model(encoder_type=args.encoder_type,
                  node_embeder=node_embeder,
                  pos_embeder=pos_embeder,
                  emb2hid=emb2hid,
                  token_embeder=token_embeder,
                  encoder=encoder,
                  decoder=decoder,
                  projector=projector,
                  init_param=args.init_param,
                  graph_emb2hid=graph_emb2hid,
                  linear_emb2hid=linear_emb2hid)
    if cuda_device is not None:
        model.to(cuda_device)
    return model


def get_gnode_value(value, mask):
    """
    value: B x N x H
    mask : B x N
    """
    batch_size, _, hid_size = value.size()
    idx = torch.sum(mask, dim=-1) - 1
    idx = idx.reshape(batch_size, 1, 1).expand(batch_size, 1, hid_size)
    return torch.gather(value, dim=1, index=idx).squeeze(1)


def get_last_value(value, mask):
    """
    value: B x N x H
    mask : B x N
    """
    batch_size, _, hid_size = value.size()
    idx = torch.sum(mask, dim=-1) - 1
    idx = idx.reshape(batch_size, 1, 1).expand(batch_size, 1, hid_size)
    return torch.gather(value, dim=1, index=idx).squeeze(1)


class Model(nn.Module):
    def __init__(self,
                 encoder_type,
                 node_embeder,
                 pos_embeder,
                 emb2hid,
                 token_embeder,
                 encoder,
                 decoder,
                 projector,
                 init_param,
                 graph_emb2hid=None,
                 linear_emb2hid=None):
        super(Model, self).__init__()
        self.encoder_type = encoder_type
        self.node_embeder = node_embeder
        self.pos_embeder = pos_embeder
        self.token_embeder = token_embeder
        self.emb2hid = emb2hid
        self.graph_emb2hid = graph_emb2hid
        self.linear_emb2hid = linear_emb2hid
        self.encoder = encoder
        self.decoder = decoder
        self.projector = projector
        if init_param:
            self._init_param()

    def _init_param(self):
        for name, param in self.named_parameters():
            if 'embeder' in name:
                continue
            elif ('weight_ih' in name) or ('weight_hh' in name):
                torch.nn.init.orthogonal_(param)
            elif param.dim() > 2:
                torch.nn.init.kaiming_uniform_(param, a=1, mode='fan_in')

    def embedding_linear_amr(self, linear_amr):
        linear_embedding = self.node_embeder(linear_amr)
        h = self.emb2hid(linear_embedding)
        return h

    def encode_linear_amr(self, h, linear_amr_mask):
        value = self.encoder(h)
        state = get_last_value(value, linear_amr_mask)
        return value, state

    def embedding_graph(self, nlabel, npos):
        nembeding = self.node_embeder(nlabel)
        pembeding = self.pos_embeder(npos)
        h = torch.cat((nembeding, pembeding), dim=-1)
        h = self.emb2hid(h)
        return h

    def encode_graph(self, adjs, relative_pos, h, node_mask):
        """Encode graph with an encoder"""
        value = self.encoder(adjs, relative_pos, h)
        state = get_gnode_value(value, node_mask)
        return value, state

    def encode_both(self, adjs, relative_pos, gh, node_mask, lh, linear_amr_mask, aligns):
        value = self.encoder(adjs, relative_pos, gh, lh, aligns)
        state = get_gnode_value(value, node_mask)
        return value, state

    def embedding_both(self, nlabel, npos, linear_amr):
        graph_embedding = self.node_embeder(nlabel)
        graph_pos_embedding = self.pos_embeder(npos)
        ge = self.graph_emb2hid(torch.cat((graph_embedding, graph_pos_embedding), dim=-1))

        linear_embedding = self.node_embeder(linear_amr)
        le = self.linear_emb2hid(linear_embedding)
        return ge, le

    def decode_tokens(self, tokens, value, value_mask, state):
        """Decode tokens with decoder and graph embeddings"""
        embeddings = self.token_embeder(tokens[:, :-1])
        decoder_outputs, attns = self.decoder(embeddings, value, value_mask, state)
        logits = self.projector(decoder_outputs)
        return logits

    def forward(self, nlabel, npos, adjs, relative_pos, node_mask, tokens, token_mask, linear_amr, linear_amr_mask, aligns):
        if self.encoder_type == 'gcn':
            h = self.embedding_graph(nlabel, npos)
            value, state = self.encode_graph(adjs, relative_pos, h, node_mask)
            logits = self.decode_tokens(tokens, value, node_mask, state)
        elif self.encoder_type == 'rnn':
            h = self.embedding_linear_amr(linear_amr)
            value, state = self.encode_linear_amr(h, linear_amr_mask)
            logits = self.decode_tokens(tokens, value, linear_amr_mask, state)
        elif self.encoder_type == 'both':
            gh, lh = self.embedding_both(nlabel, npos, linear_amr)
            value, state = self.encode_both(adjs, relative_pos, gh, node_mask, lh, linear_amr_mask, aligns)
            logits = self.decode_tokens(tokens, value, node_mask, state)

        targets = tokens[:, 1:].contiguous()
        weights = token_mask[:, 1:].float()
        loss = sequence_cross_entropy_with_logits(logits=logits, targets=targets, weights=weights)
        return loss

    def predict_with_beam_search(self, start_tokens, nlabel, npos, adjs, relative_pos, node_mask, linear_amr, linear_amr_mask, max_step, beam_size):
        if self.encoder_type == 'gcn':
            h = self.embedding_graph(nlabel, npos)
            value, state = self.encode_graph(adjs, relative_pos, h, node_mask)
            enc_mask = node_mask
        elif self.encoder_type == 'rnn':
            h = self.embedding_linear_amr(linear_amr)
            value, state = self.encoder_linear_amr(h, linear_amr_mask)
            enc_mask = linear_amr_mask
        else:
            raise NotImplementedError

        if self.decoder.config.cell_type == 'LSTM':
            c1 = torch.zeros_like(state)

        if self.decoder.config.coverage:
            cov_vec = torch.zeros(value.size(0), value.size(1))
            cov_vec = cov_vec.to(value.device)
        else:
            cov_vec = None

        history = []
        his_attns = []
        back_pointer = []

        # for the first token
        emb = self.token_embeder(start_tokens)
        if self.decoder.config.cell_type == 'GRU':
            state, cov_vec, similarity = self.decoder._step(
                emb=emb, value=value, value_mask=enc_mask,
                state=state, cov_vec=cov_vec)
        elif self.decoder.config.cell_type == 'LSTM':
            state, c1, cov_vec, similarity = self.decoder._step(
                emb=emb, value=value, value_mask=enc_mask,
                state=state, c1=c1, cov_vec=cov_vec)
        logit = self.projector(state)
        probability = torch.log_softmax(logit, dim=-1)
        log_prob_sum, t = torch.topk(probability, k=beam_size, dim=-1)           # B x bs
        history.append(t)
        his_attns.append(similarity.unsqueeze(1).repeat(1, beam_size, 1))
        t = t.view(-1)                                                    # B*bs

        value = value.repeat_interleave(beam_size, dim=0)                            # B*bs x N x H
        enc_mask = enc_mask.repeat_interleave(beam_size, dim=0)                    # B*bs x N
        state = state.repeat_interleave(beam_size, dim=0)                            # B*bs x H
        if self.decoder.config.coverage:
            cov_vec = cov_vec.repeat_interleave(beam_size, dim=0)
        else:
            cov_vec = None
        if self.decoder.config.cell_type == 'LSTM':
            c1 = c1.repeat_interleave(beam_size, dim=0)

        for idx in range(max_step - 1):
            emb = self.token_embeder(t)                                     # B*bs x E
            if self.decoder.config.cell_type == 'GRU':
                state, cov_vec, similarity = self.decoder._step(
                    emb=emb, value=value, value_mask=enc_mask,
                    state=state, cov_vec=cov_vec)
            elif self.decoder.config.cell_type == 'LSTM':
                state, c1, cov_vec, similarity = self.decoder._step(
                    emb=emb, value=value, value_mask=enc_mask,
                    state=state, c1=c1, cov_vec=cov_vec)
            his_attns.append(similarity.view(-1, beam_size, similarity.size(-1)))  # B x bs x N
            logit = self.projector(state)                           # B*bs x V
            probability = torch.log_softmax(logit, dim=-1)
            log_prob, t = torch.topk(probability, k=beam_size, dim=-1)      # B*bs x bs

            log_prob = log_prob.view(-1, beam_size * beam_size)         # B x bs*bs
            t = t.view(-1, beam_size * beam_size)               # B x bs*bs

            tmp_sum = log_prob_sum.repeat_interleave(beam_size, dim=-1)      # B x bs*bs
            tmp_sum = tmp_sum + log_prob
            log_prob_sum, pos = torch.topk(tmp_sum, beam_size, dim=-1)        # B x bs
            t = torch.gather(t, dim=-1, index=pos)              # B x bs

            hpos = torch.div(pos, beam_size)
            history.append(t)
            back_pointer.append(hpos)
            t = t.view(-1)

            hpos = hpos.unsqueeze(-1)             # B x bs x 1
            statepos = hpos.repeat_interleave(state.size(-1), dim=-1)            # B x bs x H
            state = state.view(statepos.size(0), beam_size, -1)     # B x bs x H
            state = torch.gather(state, dim=1, index=statepos)      # B x bs x H
            state = state.view(-1, state.size(-1))                  # B*bs x H
            if self.decoder.config.cell_type == 'LSTM':
                c1pos = hpos.repeat_interleave(c1.size(-1), dim=-1)
                c1 = c1.view(c1pos.size(0), beam_size, -1)
                c1 = torch.gather(c1, dim=1, index=c1pos)
                c1 = c1.view(-1, c1.size(-1))
            if self.decoder.config.coverage:
                cov_vecpos = hpos.repeat_interleave(cov_vec.size(-1), dim=-1)
                cov_vec = cov_vec.view(cov_vecpos.size(0), beam_size, -1)
                cov_vec = torch.gather(cov_vec, index=cov_vecpos, dim=1)
                cov_vec = cov_vec.view(-1, cov_vec.size(-1))

        predictions = [history[-1].cpu()]
        pred_attns = [his_attns[-1].cpu()]
        bp = None
        for i in range(max_step - 2, -1, -1):
            cur_poi = back_pointer[i].cpu()                # B x bs
            cur_his = history[i].cpu()                     # B x bs
            cur_attn = his_attns[i].cpu()                  # B x bs x N

            if bp is None:
                bp = cur_poi                         # B x bs
            else:
                bp = torch.gather(cur_poi, index=bp, dim=-1)
            cur_t = torch.gather(cur_his, index=bp, dim=-1)
            abp = bp.unsqueeze(-1)
            abp = abp.repeat(1, 1, cur_attn.size(-1))               # B x bs x N
            pred_attns.insert(0, torch.gather(cur_attn, index=abp, dim=1))
            predictions.insert(0, cur_t)
        predictions = torch.stack(predictions, dim=-1)
        attns = torch.stack(pred_attns, dim=2)
        return predictions[:, 0], attns[:, 0]

    @deprecated
    def predict(self, start_tokens, nlabel, npos, adjs, relative_pos, node_mask, max_step):
        h = self.embedding_graph(nlabel, npos)
        value, state = self.encode_graph(adjs, relative_pos, h, node_mask)
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
        attns = []
        for idx in range(max_step):
            if self.decoder.config.cell_type == 'GRU':
                state, cov_vec, similarity = self.decoder._step(
                    emb=emb, value=value, value_mask=node_mask,
                    state=state, cov_vec=cov_vec)
            elif self.decoder.config.cell_type == 'LSTM':
                state, c1, cov_vec, similarity = self.decoder._step(
                    emb=emb, value=value, value_mask=node_mask,
                    state=state, c1=c1, cov_vec=cov_vec)
            logit = self.projector(state)
            t = torch.argmax(logit, dim=-1)
            predictions.append(t)
            attns.append(similarity)
            emb = self.token_embeder(t)
        predictions = torch.stack(predictions, dim=-1)
        attns = torch.stack(attns, dim=1)
        return predictions, attns
