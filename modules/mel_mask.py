import torch
import torch.nn as nn
import torch.nn.functional as F


def random_mask(x: torch.Tensor, target_len: int):
    '''

    :param x: B T C
    :param target_len: int
    :return:
    '''
    B, T, C = x.size()
    assert T > target_len

    idx = batch_randperm(B, target_len=T, device=x.device)
    idx1, indices = torch.sort(idx[:, :target_len], dim=-1, descending=False)
    out_tensor = batch_index_select(x, idx1)
    return out_tensor, idx1


def fast_random_mask_for_cpu(x: torch.Tensor, target_len: int):
    '''

    :param x: B T C
    :param target_len: int
    :return:
    '''
    B, T, C = x.size()
    assert T > target_len

    idx = batch_randperm(B, target_len=T, device=x.device)
    idx1, indices = torch.sort(idx[:, :target_len], dim=-1, descending=False)
    out_tensor = fast_batch_index_select(x, idx1)
    return out_tensor, idx1


def fast_random_mask(x: torch.Tensor, target_len: int, masks: torch.Tensor = None):
    '''

    :param masks: B T
    :param x: B T C
    :param target_len: int
    :return:
    '''
    B, T, C = x.size()
    assert T > target_len

    idx = fast_batch_randperm(B, target_len=T, device=x.device, masks=masks)
    idx1, indices = torch.sort(idx[:, :target_len], dim=-1, descending=False)
    if masks is not None:
        masks = masks[:, :target_len]
    out_tensor = fast_batch_index_select(x, idx1, masks=masks)
    return out_tensor, idx1
def fast_random_mask_with_mask_idx(x: torch.Tensor, target_len: int, masks: torch.Tensor = None):
    '''

    :param masks: B T
    :param x: B T C
    :param target_len: int
    :return:
    '''
    B, T, C = x.size()
    assert T > target_len

    idx = fast_batch_randperm(B, target_len=T, device=x.device, masks=masks)
    idx1, indices = torch.sort(idx[:, :target_len], dim=-1, descending=False)
    mask_idx1, _ = torch.sort(idx[:, target_len:], dim=-1, descending=False)
    if masks is not None:

        masks = masks[:, :target_len]
    out_tensor = fast_batch_index_select(x, idx1, masks=masks)
    return out_tensor, idx1,mask_idx1,target_len


def batch_randperm(batch, target_len, device='cpu'):
    '''

    :param batch:
    :param target_len:
    :param device:
    :return: B T
    '''
    temp_tensor = torch.empty(batch, target_len, device=device, dtype=torch.int64)
    for i in range(batch):
        temp_tensor[i] = torch.randperm(target_len)
    return temp_tensor


def batch_index_select(x: torch.Tensor, index: torch.Tensor):
    '''

    :param x: B T C
    :param index: B T
    :return: B T C
    '''
    B, T = index.size()
    temp_tensor = torch.empty([B, T, *x.size()[2:]], dtype=x.dtype, device=x.device)

    for i in range(B):
        temp_tensor[i] = torch.index_select(x[i], 0, index[i])
    return temp_tensor

def fast_batch_randperm(batch, target_len, device='cpu', masks=None):
    '''

    :param masks: B T
    :param batch:
    :param target_len:
    :param device:
    :return: B T
    '''
    rand_idx = torch.rand(batch, target_len, device=device)
    if masks is not None:
        rand_idx = rand_idx.masked_fill(~masks, float('inf'))
    index = torch.argsort(rand_idx, dim=-1, descending=False).long()
    return index


def fast_batch_index_select(x: torch.Tensor, index: torch.Tensor, masks=None):
    '''

    :param masks: B T
    :param x: B T C
    :param index: B T
    :return: B T C
    '''
    if masks is not None:
        index = (index + 1).masked_fill(~masks, 0)
        x = torch.cat([torch.zeros(x.shape[0], 1, x.shape[2]).to(x), x], dim=1)
        output = torch.gather(x, 1, index[..., None].repeat([1, 1, x.shape[-1]]))
    else:
        output = torch.gather(x, 1, index[..., None].repeat([1, 1, x.shape[-1]]))

    return output



def re_mask(x: torch.Tensor, index: torch.Tensor, target_len: int, mask_: torch.Tensor = None):
    '''

    :param mask_: C
    :param target_len: int
    :param x: B T C
    :param index: B T
    :return: B T C
    '''
    B, _, C = x.size()

    if mask_ is not None:
        temp_tensor = mask_.repeat(B, target_len, 1)
    else:
        temp_tensor = torch.zeros(B, target_len, C, dtype=x.dtype, device=x.device)
    for i in range(B):
        temp_tensor[i][index[i]] = x[i]
    return temp_tensor


def re_mask_with_tensor(x: torch.Tensor, index: torch.Tensor, target_tensor: torch.Tensor):
    '''

    :param x: B T C
    :param index: B T
    :param target_tensor: B T1 C
    :return: B T1 C
    '''
    B, _, C = x.size()

    for i in range(B):
        target_tensor[i][index[i]] = x[i]
    return target_tensor



def fast_re_mask(x: torch.Tensor, index: torch.Tensor, target_len: int, mask_: torch.Tensor = None, masks=None):
    '''

    :param masks: B T
    :param mask_: C
    :param target_len: int
    :param x: B T C
    :param index: B T
    :return: B T C
    '''
    B, _, C = x.size()

    if mask_ is not None:
        temp_tensor = mask_.repeat(B, target_len, 1)
    else:
        temp_tensor = torch.zeros(B, target_len, C, dtype=x.dtype, device=x.device)
    batch_ind = torch.arange(B, device=x.device).unsqueeze(-1)
    if masks is not None:
        index = (index + 1).masked_fill(~masks[:, :len(index[0])], 0)
        target_tensor = torch.cat(
            [torch.zeros(temp_tensor.shape[0], 1, temp_tensor.shape[2]).to(temp_tensor), temp_tensor], dim=1)
        target_tensor[batch_ind, index] = x
        temp_tensor = target_tensor[:, 1:, :]
    else:
        temp_tensor[batch_ind, index] = x

    return temp_tensor


def fast_re_mask_with_tensor(x: torch.Tensor, index: torch.Tensor, target_tensor: torch.Tensor, masks=None):
    '''

    :param masks: B T
    :param x: B T C
    :param index: B T
    :param target_tensor: B T1 C
    :return: B T1 C
    '''
    B, _, C = x.size()

    batch_ind = torch.arange(B, device=x.device).unsqueeze(-1)
    if masks is not None:
        index = (index + 1).masked_fill(~masks[:, :len(index[0])], 0)
        target_tensor = torch.cat(
            [torch.zeros(target_tensor.shape[0], 1, target_tensor.shape[2]).to(target_tensor), target_tensor], dim=1)
        target_tensor[batch_ind, index] = x
        target_tensor = target_tensor[:, 1:, :]
    else:
        target_tensor[batch_ind, index] = x
    return target_tensor


@torch.no_grad()
def random_index_cvec(x: torch.Tensor, masks=None):
    '''

    :param masks: B T
    :param x: B T C
    :return: B T C
    '''
    B, T, C = x.size()
    x=x.detach()
    random_idx = torch.randint(0, T, size=(B, T),device=x.device)

    if masks is not None:
        mask_len = masks.long().sum(dim=1)
        mask_len = torch.unsqueeze(mask_len, 1)

        random_idx = random_idx.masked_fill(random_idx >= mask_len, 0)

    len_ind = torch.arange(T, device=x.device).unsqueeze(0)
    random_idx[random_idx == len_ind] += 1
    random_idx = random_idx.masked_fill(random_idx >= T, 0)
    random_idx[random_idx == len_ind] += 1
    if T==1:
        x=torch.cat([x, torch.randn(B, 1, C, device=x.device, dtype=x.dtype)], dim=1)
    if masks is not None:
        index = (random_idx + 1).masked_fill(~masks, 0)
        x = torch.cat([torch.zeros(x.shape[0], 1, x.shape[2]).to(x), x], dim=1)
        output = torch.gather(x, 1, index[..., None].repeat([1, 1, x.shape[-1]]))
    else:
        output = torch.gather(x, 1, random_idx[..., None].repeat([1, 1, x.shape[-1]]))

    return output


def etesst(x):
    import time
    sx = int(1024 * 0.25)
    t1 = time.time()
    ins = x
    for i in range(1000):
        out = random_mask(ins, target_len=sx)
    t2 = time.time()
    print('random_mask:', t2 - t1)
    t1 = time.time()
    for i in range(1000):
        out = fast_random_mask_for_cpu(ins, target_len=sx)
    t2 = time.time()
    print('fast_random_mask_for_cpu:', t2 - t1)
    t1 = time.time()
    for i in range(1000):
        out = fast_random_mask(ins, target_len=sx)
    t2 = time.time()
    print('fast_random_mask:', t2 - t1)
    # t1 = time.time()
    # for i in range(1000):
    #     dsdsd = Rfast_random_mask(ins, target_len=sx)
    # t2 = time.time()
    # print('Rfast_random_mask:', t2 - t1)
    t1 = time.time()
    for i in range(1000):
        eee = re_mask(out[0], index=out[1], target_len=1024)
    t2 = time.time()
    print('re_mask:', t2 - t1)
    t1 = time.time()
    for i in range(1000):
        eee = fast_re_mask(out[0], index=out[1], target_len=1024)
    t2 = time.time()
    print('fast_re_mask:', t2 - t1)
    t1 = time.time()
    for i in range(1000):
        eee = fast_re_mask_with_tensor(out[0], index=out[1], target_tensor=torch.zeros_like(x))
    t2 = time.time()
    print('fast_re_mask_with_tensor:', t2 - t1)

class MaskUtil:
    def __init__(self,configs):
        mask_type=configs['mask_arg']['mask_type']
        self.mask_type=mask_type # chunk or random
        self.mask_args=configs['mask_arg']
        if mask_type=='chunk':
            self.mask_len = self.mask_args['mask_len']
            self.mask_p = self.mask_args['mask_p']
        elif mask_type=='random':

            self.mask_p = self.mask_args['mask_p']
        else:
            raise NotImplementedError


    def chunk_mask(self,x,mask_replace_value):
        '''

        :param x: B C T
        :return:
        '''
        x=x.transpose(1,2)
        mask_replace_value=mask_replace_value.transpose(1,2)
        B, F, T = x.shape
        # mask = torch.ones(B, F, T + self.mask_len)

        time_p = (torch.rand(B, T, device=x.device) < self.mask_p).long()
        mask_indices = torch.nonzero(time_p)
        t_mask_c = torch.ones(B, T + self.mask_len, device=x.device)
        for i in mask_indices:
            t_mask_c[i[0], i[1]:i[1] + self.mask_len] = 0.
        rows = mask_indices[:, 0]
        cols = mask_indices[:, 1]
        indices_range = cols[:, None] + torch.arange(self.mask_len,device=x.device)

        # 确保不会越界
        indices_range = torch.clamp(indices_range, max=T + (self.mask_len-1))

        # 并行修改 t_mask_c 的对应位置为 0
        t_mask_c[rows[:, None], indices_range] = 0
        t_mask_c = t_mask_c[:, :-10]
        t_mask_c=t_mask_c.unsqueeze(1)
        mask_x=x*t_mask_c
        rt_mask_c=(t_mask_c==0).long()
        mask_mask_replace_value=mask_replace_value*rt_mask_c
        c_x=mask_x+mask_mask_replace_value
        c_x=c_x.transpose(1,2)
        return c_x, t_mask_c.squeeze(1)


    def random_mask(self,x,mask_replace_value):
        B, T, C = x.size()

        mask_c=torch.ones(B,T,1,device=x.device)
        ttc=int(
            T * (1 - self.mask_p) + 1)
        if ttc>=T:
            ttc=ttc-(1)
        _, un_mask_idx, mask_idx, target_len = fast_random_mask_with_mask_idx(mask_c, target_len=ttc,
                                                                                           )
        mask_c2 = fast_batch_index_select(mask_c, mask_idx)
        _, T2, _ = mask_c2.size()

        rr_mask = torch.zeros(B, T2, 1, device=x.device)

        mask_c = fast_re_mask_with_tensor(rr_mask, index=mask_idx, target_tensor=mask_c, )
        mask_x = x * mask_c
        rt_mask_c = (mask_c == 0).long()
        mask_mask_replace_value = mask_replace_value * rt_mask_c
        c_x = mask_x + mask_mask_replace_value

        return c_x, mask_c.squeeze(-1)




    def __call__(self,x,mask_replace_value):
        if self.mask_type=='chunk':
            return self.chunk_mask(x,mask_replace_value)
        elif self.mask_type=='random':
            return self.random_mask(x,mask_replace_value)
        else:
            raise NotImplementedError

if __name__ == '__main__':
    # etesst(torch.randn(2,100,48))

    mu=MaskUtil(configs={'mask_arg':{'mask_p':0.05,'mask_type':'chunk','mask_len':10}})
    x=torch.randn(2,100,48)
    x=x.fill_(10)
    x,mask=mu(x,mask_replace_value=torch.zeros_like(x))
    x=x.abs()
    pass
    mu=MaskUtil(configs={'mask_arg':{'mask_p':0.05,'mask_type':'chunk','mask_len':10}})
    x=torch.randn(2,36,48)
    x,mask=mu(x,mask_replace_value=torch.zeros_like(x))
    pass
    B, F, T = 2,10,20
    mask = torch.ones(B, F, T + 10)
    time_p = (torch.rand(B, T) < 0.1).long()
    mask_indices = torch.nonzero(time_p)
    t_mask_c = torch.ones(B, T + 10)
    # for i in mask_indices:
    #     t_mask_c[i[0], i[1]:i[1] + 10] = 0.
    rows = mask_indices[:, 0]
    cols = mask_indices[:, 1]
    indices_range = cols[:, None] + torch.arange(10)

    # 确保不会越界
    indices_range = torch.clamp(indices_range, max=T + 9)

    # 并行修改 t_mask_c 的对应位置为 0
    t_mask_c[rows[:, None], indices_range] = 0
    t_mask_c=t_mask_c[:,:-10]
    data_tensor = torch.randn(B, F, T )

    # 扩展 mask 以匹配 data_tensor 的维度
    expanded_mask = t_mask_c
    data_tensor=data_tensor.transpose(1,2)
    # 使用 masked_select 选出没有被 mask 的值（对应 mask 为 1 的部分）
    unmasked_values = data_tensor[expanded_mask == 1]
    data_tensor = torch.randn(B, T )
    unmasked_values2 = data_tensor[expanded_mask == 1]
    pass














if __name__ == '__main__':
    import time

    out = random_mask(torch.rand(1, 1024, 128), target_len=int(1024 * 0.25))
    pass

    insx = torch.rand(16, 1024, 128, device='cuda')
    # ins = torch.rand(16, 1024, 128, device='cpu')
    print('cuda_test')
    etesst(insx)
    # insx = torch.rand(16, 1024, 128, device='cuda')
    insx = torch.rand(16, 1024, 128, device='cpu')
    print('cpu_test')
    etesst(insx)
