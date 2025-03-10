

# 自己已经调试好，且理解的版本




import torch
import torch.nn as nn
import torch.nn.functional as F
# from einops import rearrange
import math

class LDConv(nn.Module):
    def __init__(self, inc, outc, num_param, stride=1, bias=None):
        super(LDConv, self).__init__()
        self.num_param = num_param
        self.stride = stride
        self.outc= outc
        self.inc = inc
        # self.conv = nn.Sequential(
        #     nn.Conv2d(inc, outc, kernel_size=(num_param, 1), stride=(num_param, 1), bias=bias),
        #     nn.BatchNorm2d(outc),
        #     nn.SiLU()
        # )
        self.conv = nn.Sequential(
            nn.Conv2d(inc*num_param,outc,kernel_size=1,stride=1, bias=bias),
            nn.BatchNorm2d(outc),
            nn.SiLU()
        )
        self.p_conv = nn.Conv2d(inc, 2 * num_param, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_full_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        offset = self.p_conv(x)
        print('offset111', offset)
        dtype = offset.data.type()
        N = offset.size(1) // 2
        p = self._get_p(offset, dtype)
        print('p111', p)
        p = p.contiguous().permute(0, 2, 3, 1)
        print('p_forward', p.shape)
        print('p222', p)


        q_lt = p.detach().floor()
        print('q_lt_forward', q_lt.shape)
        print('q_lt_forward', q_lt)
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2) - 1), torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)], dim=-1).long()
        print('q_lt_forward2', q_lt.shape)
        print('q_lt_forward2', q_lt)

        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2) - 1), torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)], dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1), torch.clamp(p[..., N:], 0, x.size(3) - 1)], dim=-1)
        print('p_forward2', p.shape)
        print('p333', p)



        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        print('q_lt_forward3', q_lt.shape)

        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        x_offset = self._reshape_x_offset(x_offset, self.num_param)
        print('x_offset',x_offset.shape)
        out = self.conv(x_offset)
        print('out', out.shape)

        return out

    def _get_p_n(self, N, dtype):
        base_int = round(math.sqrt(self.num_param))
        row_number = self.num_param // base_int
        mod_number = self.num_param % base_int
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(0, row_number),
            torch.arange(0, base_int))
        p_n_x = torch.flatten(p_n_x)
        p_n_y = torch.flatten(p_n_y)
        if mod_number > 0:
            mod_p_n_x, mod_p_n_y = torch.meshgrid(
                torch.arange(row_number, row_number + 1),
                torch.arange(0, mod_number))

            mod_p_n_x = torch.flatten(mod_p_n_x)
            mod_p_n_y = torch.flatten(mod_p_n_y)
            p_n_x, p_n_y = torch.cat((p_n_x, mod_p_n_x)), torch.cat((p_n_y, mod_p_n_y))
        p_n = torch.cat([p_n_x, p_n_y], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)
        print('P_n',p_n)
        print('P_n', p_n.shape)
        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(0, h * self.stride, self.stride),
            torch.arange(0, w * self.stride, self.stride))

        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)
        print('P_0', p_0)
        print('P_0', p_0.shape)
        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)

        p_n = self._get_p_n(N, dtype)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        # p = p_0 + p_n
        print('P', p)
        print('P', p.shape)
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        x = x.contiguous().view(b, c, -1)
        print('x', x.shape)

        index = q[..., :N] * padded_w + q[..., N:]  # offset_x*w + offset_y
        print('index', index.shape)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1)
        print('index2-1', index.shape)

        index = index.contiguous().view(b, c, -1)
        print('index2-2', index.shape)

        x_offset = x.gather(dim=-1, index=index).contiguous()
        print('x_offset_forward', x_offset.shape)
        x_offset = x_offset.view(b, c, h, w, N)
        print('x_offset_behind', x_offset.shape)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, num_param):
        b, c, h, w, n = x_offset.size()
        x_offset = x_offset.permute(0, 1, 4, 2, 3)
        x_offset = x_offset.reshape(b, c * num_param, h, w)
        return x_offset


class MyModel(nn.Module):
    def __init__(self, inc, outc, num_param):
        super(MyModel, self).__init__()
        self.conv_branch = nn.Sequential(
            nn.Conv2d(inc, outc, kernel_size=3, padding=1),
            nn.BatchNorm2d(outc),
            nn.SiLU()
        )
        self.ldconv_branch = LDConv(inc, outc, num_param)

    def forward(self, x):
        conv_out = self.conv_branch(x)
        print('1---', conv_out.shape)
        ldconv_out = self.ldconv_branch(x)
        print('2', ldconv_out.shape)
        # 合并两个分支的输出，例如简单的加权相加
        out = conv_out + ldconv_out
        print("out2", out.shape)

        return out


# 示例使用
model = MyModel(inc=32, outc=64, num_param=6)
input_data = torch.randn(1, 32, 8, 8)
output = model(input_data)
