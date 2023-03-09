'''


'''
import torch
import torch.nn as nn
import pdb

class GraphConv(nn.Module):
    """
    Graph Convolutional Network
    """
    def __init__(self, n_hops, n_users, interact_mat,
                 edge_dropout_rate=0.5, mess_dropout_rate=0.1):
        super(GraphConv, self).__init__()

        self.interact_mat = interact_mat
        self.n_users = n_users
        self.n_hops = n_hops
        self.edge_dropout_rate = edge_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate

        self.dropout = nn.Dropout(p=mess_dropout_rate)  # mess dropout

        print("卷积器不需要第0层embedding...")

    def _sparse_dropout(self, x, rate=0.5):
        noise_shape = x._nnz()

        random_tensor = rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))

    def forward(self, user_embed, item_embed,
                mess_dropout=True, edge_dropout=True):
        # user_embed: [n_users, channel]
        # item_embed: [n_items, channel]

        # all_embed: [n_users+n_items, channel]
        all_embed = torch.cat([user_embed, item_embed], dim=0)
        agg_embed = all_embed
        embs = []

        for hop in range(self.n_hops):
            interact_mat = self._sparse_dropout(self.interact_mat,
                                                self.edge_dropout_rate) if edge_dropout \
                                                                        else self.interact_mat

            agg_embed = torch.sparse.mm(interact_mat, agg_embed)
            if mess_dropout:
                agg_embed = self.dropout(agg_embed)
            # agg_embed = F.normalize(agg_embed)
            embs.append(agg_embed)
        embs = torch.stack(embs, dim=1)  # [n_entity, n_hops+1, emb_size]
        return embs[:self.n_users, :], embs[self.n_users:, :]


class LightGCN(nn.Module):
    def __init__(self, data_config, args_config, adj_mat):
        super(LightGCN, self).__init__()

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.adj_mat = adj_mat

        self.decay = args_config.l2
        self.emb_size = args_config.dim
        self.context_hops = args_config.context_hops
        self.mess_dropout = args_config.mess_dropout
        self.mess_dropout_rate = args_config.mess_dropout_rate
        self.edge_dropout = args_config.edge_dropout
        self.edge_dropout_rate = args_config.edge_dropout_rate
        self.pool = args_config.pool
        self.n_negs = args_config.n_negs
        self.ns = args_config.ns
        self.K = args_config.K

        self.device = torch.device("cuda:0") if args_config.cuda else torch.device("cpu")

        self._init_weight()
        self.user_embed = nn.Parameter(self.user_embed)
        self.item_embed = nn.Parameter(self.item_embed)

        self.gcn = self._init_model()
        print("在每一个维度上选中最难的那一个维度,构成一个item，然后继续在维度上和pos_item进行自适应权重......")

    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        self.user_embed = initializer(torch.empty(self.n_users, self.emb_size))
        self.item_embed = initializer(torch.empty(self.n_items, self.emb_size))

        # [n_users+n_items, n_users+n_items]
        self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.adj_mat).to(self.device)

    def _init_model(self):
        return GraphConv(n_hops=self.context_hops,
                         n_users=self.n_users,
                         interact_mat=self.sparse_norm_adj,
                         edge_dropout_rate=self.edge_dropout_rate,
                         mess_dropout_rate=self.mess_dropout_rate)

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def forward(self, batch=None,epoch=0):
        user = batch['users']
        pos_item = batch['pos_items']
        neg_item = batch['neg_items']  # [batch_size, n_negs * K]

        # user_gcn_emb: [n_users, channel]
        # item_gcn_emb: [n_users, channel]
        user_gcn_emb, item_gcn_emb = self.gcn(self.user_embed,
                                              self.item_embed,
                                              edge_dropout=self.edge_dropout,
                                              mess_dropout=self.mess_dropout)

        if self.ns == 'rns':  # n_negs = 1
            neg_gcn_embs = item_gcn_emb[neg_item[:, :self.K]]
        else:
            for k in range(self.K):
                neg_gcn_embs = self.negative_sampling(user_gcn_emb, item_gcn_emb,
                                                           user, neg_item[:, k*self.n_negs: (k+1)*self.n_negs],
                                                           pos_item)
            

        return self.create_bpr_loss(user_gcn_emb[user], item_gcn_emb[pos_item], neg_gcn_embs)

    def negative_sampling(self, user_gcn_emb, item_gcn_emb, user, neg_candidates, pos_item):
        batch_size = user.shape[0]
        s_e, p_e = user_gcn_emb[user], item_gcn_emb[pos_item]  # [batch_size, n_hops+1, channel]
        if self.pool != 'concat':
            s_e = self.pooling(s_e).unsqueeze(dim=1)

        # """positive mixing"""
        neg = item_gcn_emb[neg_candidates]  # [batch_size, n_negs, n_hops, channel]
        scores = s_e.unsqueeze(dim=1) * neg
        indices = torch.max(scores,1)[1].detach()
        neg = neg.permute([0, 2, 1, 3])  # [batch_size, n_hops+1, n_negs, channel]=[1,3,2,4]
        neg = neg.permute([0, 1, 3, 2])  # [batch_size, n_hops+1, channel, n_negs]=[1,3,4,2]


        # batch_size=2
        # self.n_negs=2
        # self.context_hops=3
        # self.emb_size=4
        # user = torch.randn(batch_size,self.context_hops,self.emb_size)
        # neg = torch.randn(batch_size,self.n_negs,self.context_hops,self.emb_size)  # torch.Size([1, 2, 3, 4])
        # scores = user.unsqueeze(dim=1) * neg
        # indices = torch.max(scores,dim=1)[1].detach()    # torch.Size([1, 3, 4])
        # #         indices
        # # tensor([[[0, 0, 1, 0],
        # #          [1, 0, 0, 0],
        # #          [0, 1, 1, 0]]])
        # neg = neg.permute([0, 2, 1, 3])  # [batch_size, n_hops+1, n_negs, channel]=[1,3,2,4]
        # neg = neg.permute([0, 1, 3, 2])  # [batch_size, n_hops+1, channel, n_negs]=[1,3,4,2]
        
        # # neg[0][[[i]for i in range(self.context_hops)],range(4),indices[0]]

        # neg[[[i]for i in range(batch_size)],range(self.context_hops),indices[[[j]for j in range(batch_size)],range(self.context_hops),:]]
        # neg[[[i] for i in range(batch_size)],range(neg.shape[1]), indices, :]
        # pdb.set_trace()

        _n_e_=[]
        for iii in range(batch_size):
            neg_=neg[iii]
            indices_=indices[iii]
            n_e_=neg_[[[i] for i in range(self.context_hops)],range(self.emb_size), indices_[[[i] for i in range(self.context_hops)],range(self.emb_size)]] # torch.Size([3, 4])
            _n_e_.append(n_e_)
        _n_e_=torch.stack(_n_e_,dim=0)
        


        # neg_scores = torch.abs(s_e *_n_e_)  # [batch_size, n_hops, channel]
        # total_sum = torch.abs((s_e * p_e))+neg_scores   # [batch_size, n_hops, channel]
        # total_sum=total_sum.clamp_(1e-6)

        # neg_weight = neg_scores/total_sum     # [batch_size, n_hops, channel]
        # pos_weight = 1-neg_weight   # [batch_size, n_hops, channel]
        seed = torch.rand(batch_size, p_e.shape[1], self.emb_size).to(p_e.device)  # (0, 1)
        _n_e_ = seed * p_e + (1-seed) * _n_e_  # mixing
        return _n_e_

    def pooling(self, embeddings):
        # [-1, n_hops, channel]
        if self.pool == 'mean':
            return embeddings.mean(dim=1)
        elif self.pool == 'sum':
            return embeddings.sum(dim=1)
        elif self.pool == 'concat':
            return embeddings.view(embeddings.shape[0], -1)
        else:  # final
            return embeddings[:, -1, :]

    def generate(self, split=True):
        user_gcn_emb, item_gcn_emb = self.gcn(self.user_embed,
                                              self.item_embed,
                                              edge_dropout=False,
                                              mess_dropout=False)
        user_gcn_emb, item_gcn_emb = self.pooling(user_gcn_emb), self.pooling(item_gcn_emb)
        if split:
            return user_gcn_emb, item_gcn_emb
        else:
            return torch.cat([user_gcn_emb, item_gcn_emb], dim=0)

    def rating(self, u_g_embeddings=None, i_g_embeddings=None):
        return torch.matmul(u_g_embeddings, i_g_embeddings.t())

    def create_bpr_loss(self, user_gcn_emb, pos_gcn_embs, neg_gcn_embs):
        # user_gcn_emb: [batch_size, n_hops+1, channel]
        # pos_gcn_embs: [batch_size, n_hops+1, channel]
        # neg_gcn_embs: [batch_size, n_hops+1, channel]

        batch_size = user_gcn_emb.shape[0]

        u_e = self.pooling(user_gcn_emb)
        pos_e = self.pooling(pos_gcn_embs)
        neg_e = self.pooling(neg_gcn_embs)
        # neg_e = self.pooling(neg_gcn_embs.view(-1, neg_gcn_embs.shape[2], neg_gcn_embs.shape[3])).view(batch_size, self.K, -1)

        pos_scores = torch.sum(torch.mul(u_e, pos_e), axis=1)
        neg_scores = torch.sum(torch.mul(u_e, neg_e), axis=-1)  # [batch_size, K]
        # neg_scores = torch.sum(torch.mul(u_e.unsqueeze(dim=1), neg_e), axis=-1)  # [batch_size, K]
        mf_loss = torch.mean(torch.log(1+torch.exp(neg_scores - pos_scores).sum()))

        # cul regularizer
        regularize0 = (torch.norm(user_gcn_emb[:, 0, :]) ** 2
                       + torch.norm(pos_gcn_embs[:, 0, :]) ** 2
                       + torch.norm(neg_gcn_embs[:, 0, :]) ** 2) / 2  # take hop=0
        
        regularize1 = (torch.norm(user_gcn_emb[:, 1, :]) ** 2
                       + torch.norm(pos_gcn_embs[:, 1, :]) ** 2
                       + torch.norm(neg_gcn_embs[:, 1, :]) ** 2) / 2  # take hop=0
        
        regularize2 = (torch.norm(user_gcn_emb[:, 2, :]) ** 2
                       + torch.norm(pos_gcn_embs[:, 2, :]) ** 2
                       + torch.norm(neg_gcn_embs[:,2, :]) ** 2) / 2  # take hop=0
        
        emb_loss = self.decay * (regularize0) / batch_size
        

        return mf_loss+emb_loss, mf_loss, emb_loss
        
