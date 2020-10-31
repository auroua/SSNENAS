
import torch
import torch.nn as nn
import random


class CCLNas(nn.Module):
    def __init__(self, base_encoder, input_dim=6, dim_fc=64, dim_out=32, distributed=True, train_samples=500, t=0.07,
                 min_negative_size=4500, margin=2):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(CCLNas, self).__init__()

        self.T = t

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(input_dim=input_dim, dim1=dim_fc, num_classes=dim_out)
        # create the queue
        self.distributed = distributed
        self.train_samples = train_samples

        self.min_negative_size = min_negative_size
        self.margin = margin
        self.center_list = torch.Tensor([])
        self.center_feature_list = torch.Tensor([])

    def forward(self, batch, path_encoding, device, search_space, sample_ids=None,
                logger=None):
        return self.forward_wo(batch=batch,
                               path_encoding=path_encoding,
                               sample_ids=sample_ids)

    def forward_wo(self, batch, path_encoding, sample_ids=None):
        """
        Input:
        Output:
            logits, targets
        """
        batch_nodes1, batch_edge_idx1, batch_idx1 = batch.x, batch.edge_index, batch.batch
        # compute query features
        q = self.encoder_q(batch_nodes1, batch_edge_idx1, batch_idx1)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        if sample_ids is None:
            idxs = list(range(path_encoding.size(0)))
            random.shuffle(idxs)
            sample_ids = idxs[:self.train_samples]

        logits_list = []
        center_list = []

        for i in sample_ids:
            dist_vec = torch.sum(torch.abs(path_encoding - path_encoding[i, :].view(1, -1)), dim=1)
            dist_vec[i] = 100
            min_val, _ = torch.min(dist_vec, dim=0)
            dist_vec[i] = 0
            masks = dist_vec == min_val

            negative_mask = dist_vec >= (min_val + self.margin)

            if torch.sum(negative_mask).item() < self.min_negative_size:
                continue

            posit_vecs = q[masks]
            neg_vecs = q[negative_mask][:self.min_negative_size, :]

            center = torch.mean(posit_vecs, dim=0, keepdim=True)

            center = nn.functional.normalize(center, dim=1)

            positive_pairs = torch.mm(posit_vecs, center.view(-1, 1))

            negative_pairs = torch.mm(neg_vecs, center.view(-1, 1))

            negative_pairs = negative_pairs.view(1, -1).repeat(positive_pairs.size(0), 1)

            logits = torch.cat([positive_pairs, negative_pairs], dim=1)

            logits /= self.T

            logits_list.append(logits)
            center_list.append(center.view(1, -1))

        final_logits = torch.cat(logits_list, dim=0)
        final_center = torch.cat(center_list, dim=0)
        label = torch.zeros(final_logits.shape[0], dtype=torch.long).cuda()

        return final_logits, label, final_center