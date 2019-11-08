import torch
import torch.nn as nn
import torch.distributed as dist
import torch.distributed.rpc as rpc


def _call_method(method, obj_rref, *args, **kwargs):
    return method(obj_rref.local_value(), *args, **kwargs)


def _remote_method(method, obj_rref, *args, **kwargs):
    return rpc.remote(
        obj_rref.owner(),
        _call_method,
        args=[method, obj_rref] + list(args),
        kwargs=kwargs
    )

class Encoder(nn.Module):
    def __init__(self, ntoken, ninp, dropout):
        super(Encoder, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.encoder.weight.data.uniform_(-0.1, 0.1)

    def forward(self, input):
        return self.drop(self.encoder(input))


class RNN(nn.Module):
    def __init__(self, ninp, nhid, nlayers, dropout):
        super(RNN, self).__init__()
        self.lstm = nn.LSTM(ninp, nhid, nlayers, dropout=dropout)

    def forward(self, input, hidden):
        return self.lstm(emb.to_here(), hidden)


class Decoder(nn.Module):
    def __init__(self, ntoken, nhid, dropout):
        super(Decoder, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.decoder = nn.Linear(nhid, ntoken)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-0.1, 0.1)

    def forward(self, output):
        return self.decoder(self.drop(output.to_here()))


class RNNModel(nn.Module):
    def __init__(self, ps, ntoken, ninp, nhid, nlayers, dropout=0.5):
        super(RNNModel, self).__init__()
        self.encoder_rref = rpc.remote(ps, Encoder, args=(ntoken, ninp, dropout))
        self.rnn_rref = rpc.remote(ps, RNN, args=(ninp, nhid, nlayers, dropout))
        self.decoder_rref = rpc.remote(ps, Decoder, args=(ntoken, nhid, dropout))

    def forward(self, input, hidden):
        emb_rref = _remote_method(Encoder.forward, self.encoder_rref, input)
        output_rref = _remote_method(RNN.forward, self.rnn_rref, emb_rref, hidden)
        decoded_rref = _remote_method(Decoder.forward, self.decoder_rref, output_rref)
        return decoded_rref.to_here(), hidden


@unittest.skipIf(
    not torch._six.PY3, "Pytorch distributed optim does not support python2"
)
class DistModelParallelTest(object):
    @property
    def world_size(self):
        return 4

    @property
    def init_method(self):
        return INIT_METHOD_TEMPLATE.format(
            file_name=self.file_name, rank=self.rank, world_size=self.world_size
        )

    @dist_init()
    def test_rnn(self):
        ps = 'worker%d' % ((self.rank + 1) % self.world_size)
        batch = 5
        ntoken = 10
        ninp = 20
        nhid = 30
        nlayers = 40
        rnn = RNNModel(ps, ntoken, ninp, nhid, nlayers)
        inp = torch.LongTensor()
