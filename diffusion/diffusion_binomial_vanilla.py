import torch
import torch.nn.functional as F
import numpy as np
from inspect import isfunction
from torch_scatter import scatter
import torch_geometric as pyg
from diffusion.diffusion_base import *
"""
Based in part on: https://github.com/lucidrains/denoising-diffusion-pytorch/blob/5989f4c77eafcdc6be0fb4739f0f277a6dd7f7d8/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py#L281
"""
eps = 1e-8

# 采用的离散的扩散模型，每个时间都是一个伯努利分布，对应到每个节点都是一个二项分布
# A_t = \hat{\alpha_t}A_0 + (1-\hat{\alpha_t})p
# A_t = \alpha_t A_{t-1} + (1-\alpha_t)p
# 后验分布使用贝叶斯和上面推导
class BinomialDiffusionVanilla(DiffusionBase):
    def __init__(self, num_node_classes, num_edge_classes, initial_graph_sampler, denoise_fn, timesteps=1000,
                 loss_type='vb_kl', parametrization='x0', final_prob_node=None, final_prob_edge=None, sample_time_method='importance', 
                 noise_schedule=cosine_beta_schedule, device='cuda'):
        super(BinomialDiffusionVanilla, self).__init__(num_node_classes, num_edge_classes, initial_graph_sampler, denoise_fn,
                                                    timesteps, sample_time_method, device)

        log_final_prob_node = torch.tensor(final_prob_node)[None, :].log()
        log_final_prob_edge = torch.tensor(final_prob_edge)[None, :].log()
        
        self.loss_type = loss_type
        self.parametrization = parametrization
        alphas = noise_schedule(timesteps)
        alphas = torch.tensor(alphas.astype('float64'))

        # log(alpha_1) .... log(alpha_t)
        log_alpha = np.log(alphas)
        # log(\hat(\alpha_1)) .... log(\hat{\alpha_t})
        log_cumprod_alpha = np.cumsum(log_alpha)

        # log(1-alpha_1) .... log(1-alpha_t)
        log_1_min_alpha = log_1_min_a(log_alpha)

        # log(1-\hat(\alpha_1)) .... log(1-\hat{\alpha_t})
        log_1_min_cumprod_alpha = log_1_min_a(log_cumprod_alpha)
        assert log_add_exp(log_alpha, log_1_min_alpha).abs().sum().item() < 1.e-5
        assert log_add_exp(log_cumprod_alpha, log_1_min_cumprod_alpha).abs().sum().item() < 1e-5
        assert (np.cumsum(log_alpha) - log_cumprod_alpha).abs().sum().item() < 1.e-5

        self.register_buffer('log_alpha', log_alpha.float())
        self.register_buffer('log_1_min_alpha', log_1_min_alpha.float())
        self.register_buffer('log_cumprod_alpha', log_cumprod_alpha.float())
        self.register_buffer('log_1_min_cumprod_alpha', log_1_min_cumprod_alpha.float())
        self.register_buffer('log_final_prob_node', log_final_prob_node.float())
        self.register_buffer('log_final_prob_edge', log_final_prob_edge.float())

        self.register_buffer('Lt_history', torch.zeros(timesteps))
        self.register_buffer('Lt_count', torch.zeros(timesteps))

    # 计算q(xt-1|xt, x0)正比例于q(xt-1|xt)q(xt-1|x0)
    def _q_posterior(self, log_x_start, log_x_t, t, log_final_prob):
        assert log_x_start.shape[1] == 2, f'num_class > 2 not supported'

        tmin1 = t - 1
        # Remove negative values, will not be used anyway for final decoder
        tmin1 = torch.where(tmin1 < 0, torch.zeros_like(tmin1), tmin1)
        # 通过类别来记录概率，所以存在的概率是第一列的概率
        # logp
        log_p1 = log_final_prob[:,1]
        # log(1-p)
        log_1_min_p1 = log_final_prob[:,0]

        # log(A_0)
        log_x_start_real = log_x_start[:,1]
        # log(1-A_0)
        log_1_min_x_start_real = log_x_start[:,0]

        # log(A_t)
        log_x_t_real = log_x_t[:,1]
        # log(1-A_t)
        log_1_min_x_t_real = log_x_t[:,0]


        # log(\alpha_t)
        log_alpha_t = extract(self.log_alpha, t, log_x_start_real.shape)
        # log(1- \alpha_t) = log(beta_t)
        log_beta_t = extract(self.log_1_min_alpha, t, log_x_start_real.shape)
        
        # log(\hat(\alpha_t-1))
        log_cumprod_alpha_tmin1 = extract(self.log_cumprod_alpha, tmin1, log_x_start_real.shape)
        # log(1-\hat(\alpha_t-1))
        log_1_min_cumprod_alpha_tmin1 = extract(self.log_1_min_cumprod_alpha, tmin1, log_x_start_real.shape)

        # 这个很奇怪，具体推导看https://openreview.net/pdf?id=UaAD-Nu86WX 的附录D
        # 通过变换矩阵来计算很容易，这里面的独立项相加都是变换矩阵对应的某一个变化

        log_xtmin1_eq_0_given_x_t = log_add_exp(log_beta_t+log_p1+log_x_t_real, log_1_min_a(log_beta_t+log_p1)+log_1_min_x_t_real)
        log_xtmin1_eq_1_given_x_t = log_add_exp(log_add_exp(log_alpha_t, log_beta_t+log_p1) + log_x_t_real,
                    log_beta_t + log_1_min_p1 + log_1_min_x_t_real)

        # 把整个概率看成了两部的过程，第一步是否重采样，由alpha来控制，第二步重采样的概率，由p控制
        # 计算q(A_{t-1} | A_0)
        # log(1-\hat(\alpha_t-1) * (1-p) + \hat(\alpha_t-1) * (1-A_0))
        log_xtmin1_eq_0_given_x_start = log_add_exp(log_1_min_cumprod_alpha_tmin1+log_1_min_p1, log_cumprod_alpha_tmin1 + log_1_min_x_start_real)
        # log(\hat(\alpha_t-1)*A_0 + 1-\hat(\alpha_t-1) * p1)
        log_xtmin1_eq_1_given_x_start = log_add_exp(log_cumprod_alpha_tmin1 + log_x_start_real, log_1_min_cumprod_alpha_tmin1+log_p1)

        '''
        这段文本在推导一个后验分布 $q(z_{t-1}|z_t, x)$，其中 $z_t$ 和 $z_{t-1}$ 是两个连续的隐状态，$x$ 是观察到的数据。这个推导使用了贝叶斯定理和马尔可夫性质。以下是详细的解释：

        1. **贝叶斯定理**：这是推导的开始，它表明后验分布 $q(z_{t-1}|z_t, x)$ 可以表示为 $q(z_t|z_{t-1},x) q(z_{t-1}|x)$ 的比例。这是基于贝叶斯定理的，即后验概率等于似然概率乘以先验概率除以证据（即观察到的数据的概率）。

        2. **马尔可夫性质**：这里假设噪声是马尔可夫的，即噪声在时间 $t$ 的值只依赖于在时间 $t-1$ 的值，因此 $q(z_t|z_{t-1},x) = q(z_t|z_{t-1})$。

        3. **再次应用贝叶斯定理**：这里再次应用了贝叶斯定理，得到 $q(z_t|z_{t-1}) \propto q(z_{t-1}|z_t)q(z_t)$。这里的 $\propto$ 表示比例关系，即等式左边和右边可能相差一个常数因子。

        4. **定义的使用**：根据 $Q_t$ 的定义，我们可以得到 $q(z_{t-1}|z_t) = z_t (Q_t)′$，并且由定义我们也有 $q(z_{t-1}|x) = \bar{x} Q_{t-1}$。

        5. **归一化常数**：注意到 $q(z_t)$ 并不依赖于 $z_{t-1}$，因此它可以被视为归一化常数的一部分，这是因为在计算后验分布时，我们通常会除以一个归一化常数来确保概率之和为1。

        6. **结合各项**：最后，结合以上各项，我们可以得到 $q(z_{t-1}|z_t,x) \propto z_t (Q_t)′ \odot \bar{x} Q_{t-1}$，其中 $\odot$ 表示逐元素相乘。

        这个推导的主要思想是应用贝叶斯定理和马尔可夫性质，然后利用已知的定义进行替换，最后得到了后验分布的一个表达式。'''


        log_xt_eq_0_given_xt_x_start = log_xtmin1_eq_0_given_x_t + log_xtmin1_eq_0_given_x_start
        log_xt_eq_1_given_xt_x_start = log_xtmin1_eq_1_given_x_t + log_xtmin1_eq_1_given_x_start

        unnorm_log_probs = torch.stack([log_xt_eq_0_given_xt_x_start, log_xt_eq_1_given_xt_x_start], dim=1)
        # 刚才只是正比例，最后归一化一下
        log_EV_xtmin_given_xt_given_xstart = unnorm_log_probs - unnorm_log_probs.logsumexp(1, keepdim=True)
        return log_EV_xtmin_given_xt_given_xstart

    # 去噪 一种直接预测x0，一种预测xt-1
    def _predict_x0_or_xtmin1(self, batched_graph, t_node, t_edge,embedding=None):
        out_node, out_edge = self._denoise_fn(batched_graph, t_node, t_edge,embedding)

        assert out_node.size(1) == self.num_node_classes
        assert out_edge.size(1) == self.num_edge_classes

        log_pred_node = F.log_softmax(out_node, dim=1)
        log_pred_edge = F.log_softmax(out_edge, dim=1)
        return log_pred_node, log_pred_edge

    # 本质是计算q(x_T|x_0)和p(x_T)的KL散度，或者两个分布之间的距离，变形为以下两种方式（注意两种形式都是写成要最小化的loss形式，注意符号）
    # prior matching term
    def _ce_prior(self, batched_graph):
        ones_node = torch.ones(batched_graph.nodes_per_graph.sum(), device=self.device).long()
        ones_edge = torch.ones(batched_graph.edges_per_graph.sum(), device=self.device).long()

        # 加噪到最后一步
        log_qxT_prob_node, log_qxT_prob_edge = self._q_pred(batched_graph, t_node=(self.num_timesteps - 1) * ones_node, 
                                                t_edge=(self.num_timesteps - 1) * ones_edge)

        # 默认的最后一步
        log_final_prob_node = self.log_final_prob_node * torch.ones_like(log_qxT_prob_node)
        log_final_prob_edge = self.log_final_prob_edge * torch.ones_like(log_qxT_prob_edge)

        # log_categorical就是通过交叉熵来计算 
        ce_prior_node = -log_categorical(log_qxT_prob_node, log_final_prob_node)
        ce_prior_node = scatter(ce_prior_node, batched_graph.batch, dim=-1, reduce='sum')

        ce_prior_edge = -log_categorical(log_qxT_prob_edge, log_final_prob_edge)
        ce_prior_edge = scatter(ce_prior_edge, batched_graph.batch[batched_graph.full_edge_index[0]], dim=-1, reduce='sum')

        # padding
        target_size = max(ce_prior_node.size(0), ce_prior_edge.size(0))
        if ce_prior_node.size(0)<target_size:
            pad_size=target_size-ce_prior_node.size(0)
            ce_prior_node = F.pad(ce_prior_node, (0, pad_size), "constant", 0.0000)
        elif ce_prior_edge.size(0)<target_size:
            pad_size=target_size-ce_prior_edge.size(0)
            ce_prior_edge = F.pad(ce_prior_edge, (0, pad_size), "constant", 0.0000)
        ce_prior = ce_prior_node + ce_prior_edge
        return ce_prior

    
    # prior matching term
    def _kl_prior(self, batched_graph):

        ones_node = torch.ones(batched_graph.nodes_per_graph.sum(), device=self.device).long()
        ones_edge = torch.ones(batched_graph.edges_per_graph.sum(), device=self.device).long()
        # 加噪到最后一步
        log_qxT_prob_node, log_qxT_prob_edge = self._q_pred(batched_graph, t_node=(self.num_timesteps - 1) * ones_node, 
                                                t_edge=(self.num_timesteps - 1) * ones_edge)
        # 默认的最后一步
        log_final_prob_node = self.log_final_prob_node * torch.ones_like(log_qxT_prob_node)
        log_final_prob_edge = self.log_final_prob_edge * torch.ones_like(log_qxT_prob_edge)

        # 这个是正经的交叉熵
        kl_prior_node = self.multinomial_kl(log_qxT_prob_node, log_final_prob_node)
        kl_prior_node = scatter(kl_prior_node, batched_graph.batch, dim=-1, reduce='sum')

        kl_prior_edge = self.multinomial_kl(log_qxT_prob_edge, log_final_prob_edge)
        kl_prior_edge = scatter(kl_prior_edge, batched_graph.batch[batched_graph.full_edge_index[0]], dim=-1, reduce='sum')

        kl_prior = kl_prior_node + kl_prior_edge

        return kl_prior
   
   # 不计算x_t-1的分布，直接采样x_t-1，计算KL散度
   # 将KL散度分解成了交叉熵和熵的和，使用monte karlo来估计复杂的KL散度，通俗的理解就是加噪过程步骤要符合样本，去噪过程要符合样本，而不是直接让去噪过程符合真实的分布
    def _compute_MC_KL(self, batched_graph, t_edge, t_node):
        # print("batched_graph:",batched_graph)
        # 去噪之后的p(xt-1|xt)
        log_model_prob_node, log_model_prob_edge = self._p_pred(batched_graph=batched_graph, t_node=t_node, t_edge=t_edge)
        # 计算q(x_t|x_{t-1})
        log_true_prob_node, log_true_prob_edge = self._q_pred_one_timestep(batched_graph=batched_graph, t_node=t_node, t_edge=t_edge)

        # 直接根据采样得到的x_{t-1}来计算
        cross_ent_node = -log_categorical(batched_graph.log_node_attr_tmin1, log_model_prob_node)
        cross_ent_edge = -log_categorical(batched_graph.log_full_edge_attr_tmin1, log_model_prob_edge)

        # 直接根据从x_t-1预测xt概率，采样得到的x_{t}来计算
        ent_node = log_categorical(batched_graph.log_node_attr_t, log_true_prob_node).detach()
        ent_edge = log_categorical(batched_graph.log_full_edge_attr_t, log_true_prob_edge).detach()

        loss_node = cross_ent_node + ent_node
        loss_edge = cross_ent_edge + ent_edge

        loss_node = scatter(loss_node, batched_graph.batch, dim=-1, reduce='sum')
        loss_edge = scatter(loss_edge, batched_graph.batch[batched_graph.full_edge_index[0]], dim=-1, reduce='sum') 
        # 填充
        target_size = max(loss_node.size(0), loss_edge.size(0))
        if loss_node.size(0)<target_size:
            pad_size=target_size-loss_node.size(0)
            loss_node = F.pad(loss_node, (0, pad_size), "constant", 0.0000)
        elif loss_edge.size(0)<target_size:
            pad_size=target_size-loss_edge.size(0)
            loss_edge = F.pad(loss_edge, (0, pad_size), "constant", 0.0000)
        loss = loss_node + loss_edge
        return loss

    # 只需要采样最后的x_t,结合最开始的x0，计算出x_t-1的分布
    def _compute_RB_KL(self, batched_graph, t, t_edge, t_node):
        # 计算q(xt-1|xt, x0)
        log_true_prob_node = self._q_posterior(log_x_start=batched_graph.log_node_attr, 
                                            log_x_t=batched_graph.log_node_attr_t, t=t_node, log_final_prob=self.log_final_prob_node)
        log_true_prob_edge = self._q_posterior(log_x_start=batched_graph.log_full_edge_attr, 
                                            log_x_t=batched_graph.log_full_edge_attr_t, t=t_edge, log_final_prob=self.log_final_prob_edge)
        # 去噪之后的p(xt-1|xt)
        log_model_prob_node, log_model_prob_edge = self._p_pred(batched_graph=batched_graph, t_node=t_node, t_edge=t_edge) 

        # 跟现有公示不太一样，得确认一下
        # 计算denoising matching term
        kl_node = self.multinomial_kl(log_true_prob_node, log_model_prob_node)
        kl_node = scatter(kl_node, batched_graph.batch, dim=-1, reduce='sum')
        kl_edge = self.multinomial_kl(log_true_prob_edge, log_model_prob_edge)
        kl_edge = scatter(kl_edge, batched_graph.batch[batched_graph.full_edge_index[0]], dim=-1, reduce='sum')
        kl = kl_node + kl_edge

        # 这个是计算reconstruction term，通过monte karlo计算
        # log_node_attr就是node特征的log，node特征默认为0
        decoder_nll_node = -log_categorical(batched_graph.log_node_attr, log_model_prob_node)
        decoder_nll_node = scatter(decoder_nll_node, batched_graph.batch, dim=-1, reduce='sum')
        # log_full_edge_attr就是edge特征的log，edge存在为1，不存在为0
        # 因为第一步的变化特别少，所以q(x1|x0)，近似于q(x0)，所以直接用原始分布来计算期望
        decoder_nll_edge = -log_categorical(batched_graph.log_full_edge_attr, log_model_prob_edge)
        decoder_nll_edge = scatter(decoder_nll_edge, batched_graph.batch[batched_graph.full_edge_index[0]], dim=-1, reduce='sum')
        decoder_nll = decoder_nll_node + decoder_nll_edge

        mask = (t == torch.zeros_like(t)).float()
        loss = mask * decoder_nll + (1. - mask) * kl

        return loss

    # 采样并设置q(xt|x0) 
    def _q_sample_and_set_xt_given_x0(self, batched_graph, t_node, t_edge):
        batched_graph.log_node_attr = index_to_log_onehot(batched_graph.node_attr, self.num_node_classes)
        batched_graph.log_full_edge_attr = index_to_log_onehot(batched_graph.full_edge_attr, self.num_edge_classes)
        
        log_node_attr_t, log_full_edge_attr_t = self.q_sample(batched_graph, t_node, t_edge)
        
        batched_graph.log_node_attr_t = log_node_attr_t
        batched_graph.log_full_edge_attr_t = log_full_edge_attr_t 
        
    # 采样并设置q(xt|xt-1)  q(xt-1|x0)
    def _q_sample_and_set_xtmin1_xt_given_x0(self, batched_graph, t_node, t_edge):
        batched_graph.log_node_attr = index_to_log_onehot(batched_graph.node_attr, self.num_node_classes)
        batched_graph.log_full_edge_attr = index_to_log_onehot(batched_graph.full_edge_attr, self.num_edge_classes)
        
        # sample xt-1
        tmin1_node = t_node - 1
        tmin1_edge = t_edge - 1
        tmin1_node_clamped = torch.where(tmin1_node < 0, torch.zeros_like(tmin1_node), tmin1_node)
        tmin1_edge_clamped = torch.where(tmin1_edge < 0, torch.zeros_like(tmin1_edge), tmin1_edge)
        
        log_node_attr_tmin1, log_full_edge_attr_tmin1 = self.q_sample(batched_graph, tmin1_node_clamped, tmin1_edge_clamped)
        batched_graph.log_node_attr_tmin1 = log_node_attr_tmin1
        batched_graph.log_full_edge_attr_tmin1 = log_full_edge_attr_tmin1

        batched_graph.log_node_attr_tmin1[tmin1_node<0] = batched_graph.log_node_attr[tmin1_node<0]
        batched_graph.log_full_edge_attr_tmin1[tmin1_edge<0] = batched_graph.log_full_edge_attr[tmin1_edge<0]

        # sample xt given xt-1
        log_node_attr_t, log_full_edge_attr_t = self._q_sample_one_timestep(batched_graph, t_node, t_edge)
        batched_graph.log_node_attr_t = log_node_attr_t
        batched_graph.log_full_edge_attr_t = log_full_edge_attr_t


    # 采样q(xt|xt-1)
    def _q_sample_one_timestep(self, batched_graph, t_node, t_edge):
        log_prob_node, log_prob_edge = self._q_pred_one_timestep(batched_graph, t_node, t_edge)

        log_out_node = self.log_sample_categorical(log_prob_node, self.num_node_classes)

        log_out_edge = self.log_sample_categorical(log_prob_edge, self.num_edge_classes)

        return log_out_node, log_out_edge  

        # xt ~ q(xt|xtmin1)

    # 计算q(xt|xt-1) 
    def _q_pred_one_timestep(self, batched_graph, t_node, t_edge):
        # log(\alpha_t)
        log_alpha_t_node = extract(self.log_alpha, t_node, batched_graph.log_node_attr.shape)
        # log(1- \alpha_t) = log(beta_t)
        log_1_min_alpha_t_node = extract(self.log_1_min_alpha, t_node, batched_graph.log_node_attr.shape)

        # 计算q(x_t|x_{t-1})
        # log(x_{t-1}*alpha_t + final的概率*(1-\alpha_t))
        log_prob_nodes = log_add_exp(
            batched_graph.log_node_attr_tmin1 + log_alpha_t_node,
            log_1_min_alpha_t_node + self.log_final_prob_node
        )

        log_alpha_t_edge = extract(self.log_alpha, t_edge, batched_graph.log_full_edge_attr.shape)
        log_1_min_alpha_t_edge = extract(self.log_1_min_alpha, t_edge, batched_graph.log_full_edge_attr.shape)

        log_prob_edges = log_add_exp(
            batched_graph.log_full_edge_attr_tmin1 + log_alpha_t_edge,
            log_1_min_alpha_t_edge + self.log_final_prob_edge
        )

        return log_prob_nodes, log_prob_edges 

    def _calc_num_entries(self, batched_graph):
        return batched_graph.full_edge_attr.shape[0] + batched_graph.node_attr.shape[0]

    # 采样t
    def _sample_time(self, b, device, method='uniform'):
        if method == 'importance':
            if not (self.Lt_count > 10).all():
                return self._sample_time(b, device, method='uniform')

            Lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 0.0001
            Lt_sqrt[0] = Lt_sqrt[1]  # Overwrite decoder term with L1.
            pt_all = Lt_sqrt / Lt_sqrt.sum()

            t = torch.multinomial(pt_all, num_samples=b, replacement=True)

            pt = pt_all.gather(dim=0, index=t)
            return t, pt

        elif method == 'uniform':
            length = self.Lt_count.shape[0]
            # 采样长度1000的话是0-999
            t = torch.randint(0, length, (b,), device=device).long()

            pt = torch.ones_like(t).float() / length
            return t, pt
        else:
            raise ValueError 
    
    # 计算q(xt|x0) 
    def _q_pred(self, batched_graph, t_node, t_edge):
        # nodes prob
        log_cumprod_alpha_t_node = extract(self.log_cumprod_alpha, t_node, batched_graph.log_node_attr.shape)
        log_1_min_cumprod_alpha_node = extract(self.log_1_min_cumprod_alpha, t_node, batched_graph.log_node_attr.shape)
        log_prob_nodes = log_add_exp(
            batched_graph.log_node_attr + log_cumprod_alpha_t_node, 
            log_1_min_cumprod_alpha_node + self.log_final_prob_node 
        ) 

        # edges prob
        log_cumprod_alpha_t_edge = extract(self.log_cumprod_alpha, t_edge, batched_graph.log_full_edge_attr.shape)
        log_1_min_cumprod_alpha_edge = extract(self.log_1_min_cumprod_alpha, t_edge, batched_graph.log_full_edge_attr.shape)
        log_prob_edges = log_add_exp(
            batched_graph.log_full_edge_attr + log_cumprod_alpha_t_edge,
            log_1_min_cumprod_alpha_edge + self.log_final_prob_edge
        )
        return log_prob_nodes, log_prob_edges 

    # 计算p(xt-1|xt) 两种模式，一种直接预测x0，一种预测xt-1
    def _p_pred(self, batched_graph, t_node, t_edge,embedding=None):
        if self.parametrization == 'x0':
            # 预测x0
            log_node_recon, log_full_edge_recon = self._predict_x0_or_xtmin1(batched_graph, t_node=t_node, t_edge=t_edge,embedding=embedding)
            # 基于这个x0来通过q(xt-1|xt, x0)再计算xt-1
            log_model_pred_node = self._q_posterior(
                log_x_start=log_node_recon, log_x_t=batched_graph.log_node_attr_t, t=t_node, log_final_prob=self.log_final_prob_node)
            log_model_pred_edge = self._q_posterior(
                log_x_start=log_full_edge_recon, log_x_t=batched_graph.log_full_edge_attr_t, t=t_edge, log_final_prob=self.log_final_prob_edge)
        elif self.parametrization == 'xt':
            # 直接预测x_{t-1}
            log_model_pred_node, log_model_pred_edge = self._predict_x0_or_xtmin1(batched_graph, t_node=t_node, t_edge=t_edge,embedding=embedding) 
        # print("_p_pred-log_model_pred_node, log_model_pred_edge:",log_model_pred_node, log_model_pred_edge)
        return log_model_pred_node, log_model_pred_edge


    def _prepare_data_for_sampling(self, batched_graph):
        batched_graph.log_node_attr = index_to_log_onehot(batched_graph.node_attr, self.num_node_classes)
        log_prob_node = torch.ones_like(batched_graph.log_node_attr, device=self.device) * self.log_final_prob_node
        batched_graph.log_node_attr_t = self.log_sample_categorical(log_prob_node, self.num_node_classes)
        

        batched_graph.log_full_edge_attr = index_to_log_onehot(batched_graph.full_edge_attr, self.num_edge_classes)

        log_prob_edge = torch.ones_like(batched_graph.log_full_edge_attr, device=self.device) * self.log_final_prob_edge

      
        batched_graph.log_full_edge_attr_t = self.log_sample_categorical(log_prob_edge, self.num_edge_classes)

        return batched_graph

    def _eval_loss(self, batched_graph):
        b = batched_graph.num_graphs
        batched_graph.num_entries = self._calc_num_entries(batched_graph)
        if self.loss_type == 'vb_kl':
            t, pt = self._sample_time(b, self.device, self.sample_time_method)
            
            t_node = t.repeat_interleave(batched_graph.nodes_per_graph)
            t_edge = t.repeat_interleave(batched_graph.edges_per_graph)
            
            self._q_sample_and_set_xt_given_x0(batched_graph, t_node, t_edge)

            kl = self._compute_RB_KL(batched_graph, t, t_edge, t_node,embedding=None)
            kl_prior = self._kl_prior(batched_graph=batched_graph)
            # Upweigh loss term of the kl
            loss = kl / pt + kl_prior
            return -loss
        
        elif self.loss_type == 'vb_ce_x0':
            assert self.parametrization == 'x0'
            pass #TODO not in the scope of the current submission

        elif self.loss_type == 'vb_ce_xt':
            assert self.parametrization == 'xt'

            t, pt =  self._sample_time(b, self.device, self.sample_time_method)

            t_node = t.repeat_interleave(batched_graph.nodes_per_graph)
            t_edge = t.repeat_interleave(batched_graph.edges_per_graph)

            self._q_sample_and_set_xtmin1_xt_given_x0(batched_graph, t_node, t_edge)

            kl = self._compute_MC_KL(batched_graph, t_edge, t_node)

            ce_prior = self._ce_prior(batched_graph=batched_graph)
            # Upweigh loss term of the kl
            vb_loss = kl / pt + ce_prior
            return -vb_loss 

       
        else:
            raise ValueError()


    def _train_loss(self, batched_graph):
        b = batched_graph.num_graphs
        batched_graph.num_entries = self._calc_num_entries(batched_graph) 
        if self.loss_type == 'vb_kl':             
            # not sure it is ok to allow the parameterization to be xt, which is also sensible in math, for now it must be x0
            assert self.parametrization == 'x0'
            # sample t for each graph
            t, pt = self._sample_time(b, self.device, self.sample_time_method)

            t_node = t.repeat_interleave(batched_graph.nodes_per_graph)
            t_edge = t.repeat_interleave(batched_graph.edges_per_graph)
            
            self._q_sample_and_set_xt_given_x0(batched_graph, t_node, t_edge)

            kl = self._compute_RB_KL(batched_graph, t, t_edge, t_node)

            Lt2 = kl.pow(2)
            Lt2_prev = self.Lt_history.gather(dim=0, index=t)
            new_Lt_history = (0.1 * Lt2 + 0.9 * Lt2_prev).detach()
            self.Lt_history.scatter_(dim=0, index=t, src=new_Lt_history)
            self.Lt_count.scatter_add_(dim=0, index=t, src=torch.ones_like(Lt2))

            kl_prior = self._kl_prior(batched_graph=batched_graph)

            # Upweigh loss term of the kl
            vb_loss = kl / pt + kl_prior
            return -vb_loss
        
        
        elif self.loss_type == 'vb_ce_x0':
            assert self.parametrization == 'x0'
            pass # TODO 


        elif self.loss_type == 'vb_ce_xt':
            assert self.parametrization == 'xt'

            t, pt =  self._sample_time(b, self.device, self.sample_time_method)

            t_node = t.repeat_interleave(batched_graph.nodes_per_graph)
            t_edge = t.repeat_interleave(batched_graph.edges_per_graph)
            
            self._q_sample_and_set_xtmin1_xt_given_x0(batched_graph, t_node, t_edge)


            kl = self._compute_MC_KL(batched_graph, t_edge, t_node)

            Lt2 = kl.pow(2)
            Lt2_prev = self.Lt_history.gather(dim=0, index=t)
            new_Lt_history = (0.1 * Lt2 + 0.9 * Lt2_prev).detach()
            self.Lt_history.scatter_(dim=0, index=t, src=new_Lt_history)
            self.Lt_count.scatter_add_(dim=0, index=t, src=torch.ones_like(Lt2))

            ce_prior = self._ce_prior(batched_graph=batched_graph)
            # Upweigh loss term of the kl
            vb_loss = kl / pt + ce_prior

            return -vb_loss 


        else:
            raise ValueError()
    