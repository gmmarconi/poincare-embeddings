#!/usr/bin/env python3
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
from numpy.random import choice, randint
import torch as th
from kernelregressionutils import sqdist
from torch import nn
from torch.autograd import Function, Variable
from torch.utils.data import Dataset
from collections import defaultdict as ddict

eps = 1e-5


class Arcosh(Function):
    def __init__(self, eps=eps):
        super(Arcosh, self).__init__()
        self.eps = eps

    def forward(self, x):
        self.z = th.sqrt(x * x - 1)
        return th.log(x + self.z)

    def backward(self, g):
        z = th.clamp(self.z, min=eps)
        z = g / z
        return z


class PoincareDistance(Function):
    boundary = 1 - eps

    def grad(self, x, v, sqnormx, sqnormv, sqdist):
        alpha = (1 - sqnormx)
        beta = (1 - sqnormv)
        z = 1 + 2 * sqdist / (alpha * beta)
        a = ((sqnormv - 2 * th.sum(x * v, dim=-1) + 1) / th.pow(alpha, 2)).unsqueeze(-1).expand_as(x)
        a = a * x - v / alpha.unsqueeze(-1).expand_as(v)
        z = th.sqrt(th.pow(z, 2) - 1)
        z = th.clamp(z * beta, min=eps).unsqueeze(-1)
        return 4 * a / z.expand_as(x)

    def forward(self, u, v):
        self.save_for_backward(u, v)
        self.squnorm = th.clamp(th.sum(u * u, dim=-1), 0, self.boundary)
        self.sqvnorm = th.clamp(th.sum(v * v, dim=-1), 0, self.boundary)
        self.sqdist = th.sum(th.pow(u - v, 2), dim=-1)
        x = self.sqdist / ((1 - self.squnorm) * (1 - self.sqvnorm)) * 2 + 1
        # arcosh
        z = th.sqrt(th.pow(x, 2) - 1)
        return th.log(x + z)

    def backward(self, g):
        u, v = self.saved_tensors
        g = g.unsqueeze(-1)
        gu = self.grad(u, v, self.squnorm, self.sqvnorm, self.sqdist)
        gv = self.grad(v, u, self.sqvnorm, self.squnorm, self.sqdist)
        return g.expand_as(gu) * gu, g.expand_as(gv) * gv


class EuclideanDistance(nn.Module):
    def __init__(self, radius=1, dim=None):
        super(EuclideanDistance, self).__init__()

    def forward(self, u, v):
        return th.sum(th.pow(u - v, 2), dim=-1)


class TranseDistance(nn.Module):
    def __init__(self, radius=1, dim=None):
        super(TranseDistance, self).__init__()
        self.r = nn.Parameter(th.randn(dim).view(1, dim))

    def forward(self, u, v):
        # batch mode
        if u.dim() == 3:
            r = self.r.unsqueeze(0).expand(v.size(0), v.size(1), self.r.size(1))
        # non batch
        else:
            r = self.r.expand(v.size(0), self.r.size(1))
        return th.sum(th.pow(u - v + r, 2), dim=-1)


class Embedding(nn.Module):
    def __init__(self, size, dim, dist=PoincareDistance, max_norm=1):
        super(Embedding, self).__init__()
        self.dim = dim
        self.lt = nn.Embedding(
            size, dim,
            max_norm=max_norm,
            sparse=True,
            scale_grad_by_freq=False
        )
        self.dist = dist
        self.init_weights()

    def init_weights(self, scale=1e-4):
        self.lt.state_dict()['weight'].uniform_(-scale, scale)

    def forward(self, inputs):
        e = self.lt(inputs)
        fval = self._forward(e)
        return fval

    def embedding(self):
        return list(self.lt.parameters())[0].data.cpu().numpy()


class SNEmbedding(Embedding):
    def __init__(self, size, dim, dist=PoincareDistance, max_norm=1):
        super(SNEmbedding, self).__init__(size, dim, dist, max_norm)
        self.lossfn = nn.CrossEntropyLoss

    def _forward(self, e):
        o = e.narrow(1, 1, e.size(1) - 1)
        s = e.narrow(1, 0, 1).expand_as(o)
        dists = self.dist()(s, o).squeeze(-1)
        return -dists

    def loss(self, preds, targets, weight=None, size_average=True):
        lossfn = self.lossfn(size_average=size_average, weight=weight)
        return lossfn(preds, targets)


class GraphDataset(Dataset):
    _ntries = 100
    _dampening = 1

    def __init__(self, idx, objects, nnegs, unigram_size=1e8):
        print('Indexing data')
        self.idx = idx
        self.nnegs = nnegs
        self.burnin = False
        self.objects = objects
        self.max_tries = self.nnegs * self._ntries

        self._weights = ddict(lambda: ddict(int))   # default int is 0
        self._counts = np.ones(len(objects), dtype=np.float)
        for i in range(idx.size(0)):
            t, h, w = [int(x) for x in self.idx[i]]
            self._counts[h] += w
            self._weights[t][h] += w
        self._weights = dict(self._weights) # dict of dict, each subdict represents an item (indexed by "objects") and contains the PARENTS


        if unigram_size > 0:
            c = self._counts ** self._dampening
            self.unigram_table = choice(
                len(objects),
                size=int(unigram_size),
                p=(c / c.sum())
            )

    def __len__(self):
        return self.idx.size(0)

    @classmethod
    def collate(cls, batch):
        inputs, targets = zip(*batch)
        return Variable(th.cat(inputs, 0)), Variable(th.cat(targets, 0))


class SNGraphDataset(GraphDataset):
    model_name = '%s_%s_dim%d'

    def __getitem__(self, i):
        t, h, _ = [int(x) for x in self.idx[i]]
        t = int(t)
        h = int(h)
        negs = set()
        ntries = 0
        nnegs = self.nnegs
        if self.burnin:
            nnegs *= 0.1
        while ntries < self.max_tries and len(negs) < nnegs:
            if self.burnin:
                n = randint(0, len(self.unigram_table))
                n = int(self.unigram_table[n])
            else:
                n = randint(0, len(self.objects))
            if n not in self._weights[t]:
                negs.add(n)
            ntries += 1
        if len(negs) == 0:
            negs.add(t)
        ix = [t, h] + list(negs)
        while len(ix) < nnegs + 2:
            ix.append(ix[randint(2, len(ix))])
        return th.LongTensor(ix).view(1, len(ix)), th.zeros(1).long()

    @classmethod
    def initialize(cls, distfn, opt, idx, objects, max_norm=1):
        conf = []
        model_name = cls.model_name % (opt.dset, opt.distfn, opt.dim)
        data = cls(idx, objects, opt.negs) # istantiates a GraphDataset object
        model = SNEmbedding(
            len(data.objects),
            opt.dim,
            dist=distfn,
            max_norm=max_norm
        )
        data.objects = objects
        return model, data, model_name, conf


class GraphDatasetSupervised(Dataset):
    _ntries = 10
    _dampening = 1

    def __init__(self, idx, objects, xtr, nnegs, unigram_size=1e8, similarity="gram", margin=False):
        print('Indexing data')
        self.idx = idx
        self.nnegs = nnegs
        self.burnin = False
        self.objects = objects
        self.max_tries = self.nnegs * self._ntries

        # Supervised info
        self.xtr = xtr
        self.nlabels = sum(x['feature'] == -1 for x in objects)
        if self.nlabels == 0:
            self.nlabels = len(objects)
        if similarity is "gram":
            print("Computing Gram matrix")
            self.gramian = self.xtr.dot(self.xtr)
            self.gramian_ord_idx = np.argsort(self.gramian, axis=1)
        elif similarity is "distance":
            self.gramian = sqdist(self.xtr, self.xtr)
            self.gramian_ord_idx = np.argsort(-self.gramian, axis=1)

        self._weights = ddict(lambda: ddict(int))  # default int is 0, weights of edges (usually equal to 1)
        self._counts = np.ones(len(objects), dtype=np.float)
        print("Creating weights list")
        for i in range(idx.size(0)):
            t, h, w = [int(x) for x in self.idx[i]]
            self._counts[h] += w
            self._weights[t][h] += w
        self._weights = dict(self._weights)


        if unigram_size > 0:
            c = self._counts ** self._dampening
            self.unigram_table = choice(
                len(objects),
                size=int(unigram_size),
                p=(c / c.sum())
            )

    def __len__(self):
        return self.idx.size(0)

    @classmethod
    def collate(cls, batch):
        inputs, targets = zip(*batch)
        return Variable(th.cat(inputs, 0)), Variable(th.cat(targets, 0))


class SNGraphDatasetSupervised(GraphDatasetSupervised):
    model_name = '%s_%s_dim%d'

    def __getitem__(self, i):

        t, h, _ = [int(x) for x in self.idx[i]]  # t: child, h:parent

        # t corresponds to a label
        if self.objects[t]['feature'] == -1 or self.checkIfleastSimilarL(t, h):
            negs = set()
            ntries = 0
            nnegs = self.nnegs
            if self.burnin:
                nnegs *= 0.1
            while ntries < self.max_tries and len(negs) < nnegs:
                if self.burnin:
                    n = randint(0, len(self.unigram_table))
                    n = int(self.unigram_table[n])
                else:
                    n = randint(0, self.nlabels) # check if in self_weights
                if n not in self._weights[t]:   # does not contain all indexes
                    negs.add(n)
                ntries += 1
            if len(negs) == 0:
                negs.add(t)
            ix = [t, h] + list(negs)
            while len(ix) < nnegs + 2:
                ix.append(ix[randint(2, len(ix))])
            return th.LongTensor(ix).view(1, len(ix)), th.zeros(1).long()
        # t corresponds to an instance
        elif self.objects[t]['feature'] >= 0:
            t, h, _ = [int(x) for x in self.idx[i]] #t: child, h:parent
            negs = set()
            ntries = 0
            nnegs = self.nnegs
            if self.burnin:
                nnegs *= 0.1
            while ntries < self.max_tries and len(negs) < nnegs:
                if self.burnin:
                    n = randint(0, len(self.unigram_table))
                    less_odx = int(self.unigram_table[n])
                else:
                    less_odx = self.lessSimilarThan(child_idxo=t, parent_idxo=h)
                if less_odx not in self._weights[t]:
                    negs.add(less_odx)
                ntries += 1
            if len(negs) == 0:
                negs.add(t)
            ix = [t, h] + list(negs)
            while len(ix) < nnegs + 2:
                ix.append(ix[randint(2, len(ix))])
            return th.LongTensor(ix).view(1, len(ix)), th.zeros(1).long()

        else:
            print("SNGraphDatasetsupervised>> Error, requested item is not a label nor an instance")
            return False

    @classmethod
    def initialize(cls, distfn, opt, idx, objects, max_norm=1, xtr=None, similarity='gram'):
        if xtr is None:
            print("Please supply a feature matrix")
        conf = []
        model_name = cls.model_name % (opt.dset, opt.distfn, opt.dim)
        data = cls(idx, objects, xtr, opt.negs, similarity=similarity)  # istantiates a GraphDataset object
        model = SNEmbedding(
            len(data.objects),
            opt.dim,
            dist=distfn,
            max_norm=max_norm
        )
        data.objects = objects
        return model, data, model_name, conf

    def feature2objectsidx(self, idxf):
        """Returns the 'objects' index associated to the 'feature' index"""
        return [i for (i, v) in enumerate(self.objects) if v['feature'] ==  idxf][0]

    def objects2featureidx(self, idxo):
        """Returns the 'feature' index associated to the 'feature' index"""
        return self.objects[idxo]['feature']

    def lessSimilarThan(self, child_idxo, parent_idxo):
        """Returns a node less similar to parent than child

        :child: objects idx of the child
        :parent: objects idx of the parent
        """
        parent_idxf = self.objects2featureidx(parent_idxo)
        child_idxf = self.objects2featureidx(child_idxo)
        parent_similarity_index = int(np.argwhere(parent_idxf == self.gramian_ord_idx[child_idxf,:])[0])
        if parent_similarity_index < 1:
            print("lessSimilarThan(): the least similar, parent_idxf: %d\t child_idxf: %d" % (parent_idxf, child_idxf))
        less_fidx = np.random.choice(self.gramian_ord_idx[child_idxf,:parent_similarity_index])
        less_odx = [i for (i, v) in enumerate(self.objects) if v['feature'] == less_fidx][0]
        return less_odx

    def checkIfleastSimilarL(self, child_idxo, parent_idxo):
        """Checks if the parent is the least similar node in feature space"""
        parent_idxf = self.objects2featureidx(parent_idxo)
        child_idxf = self.objects2featureidx(child_idxo)
        parent_similarity_index = int(np.argwhere(parent_idxf == self.gramian_ord_idx[child_idxf,:])[0])
        if parent_similarity_index < 1:
            return True
        else:
            return False
