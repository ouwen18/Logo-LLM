import torch
import torch.nn as nn
from modelscope import AutoModel
from einops import rearrange
from transformers.models.gpt2.configuration_gpt2 import GPT2Config

class LogoLLM(nn.Module):
  def __init__(self, configs, device):
    super(LogoLLM, self).__init__()
    self.is_gpt = configs.is_gpt
    self.patch_size = configs.patch_size
    self.pretrain = configs.pretrain
    self.stride = configs.stride
    self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 1

    self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
    self.patch_num += 1

    if configs.is_gpt:
      if configs.pretrain:
        self.gpt2 = AutoModel.from_pretrained('openai-community/gpt2', output_attentions=True,
                                              output_hidden_states=True)  # loads a pretrained GPT-2 base model
      else:
        print("------------------no pretrain------------------")
        self.gpt2 = AutoModel(GPT2Config())
      self.gpt2.h = self.gpt2.h[:configs.gpt_layers]
      print("gpt2 = {}".format(self.gpt2))

    self.version = configs.version
    self.in_layer = nn.Linear(configs.patch_size, configs.d_model)
    self.out_layer = nn.Linear(configs.d_model * self.patch_num, configs.pred_len)


    for i, (name, param) in enumerate(self.gpt2.named_parameters()):
      if 'ln' in name or 'wpe' in name:
        param.requires_grad = True
      else:
        param.requires_grad = False

    for layer in (self.gpt2, self.in_layer, self.out_layer):
      layer.to(device=device)
      layer.train()

    self.mixer = Mixer(configs, device).to(device)
    self.num_layer = configs.gpt_layers // 2


  def forward(self, x, itr):
    B, L, M = x.shape

    ## instance normalization
    means = x.mean(1, keepdim=True).detach()
    x = x - means
    stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
    x /= stdev

    x = rearrange(x, 'b l m -> b m l')

    ## patch
    x = self.padding_patch_layer(x)
    x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
    x = rearrange(x, 'b m n p -> (b m) n p')

    ## linear embedding
    x_embed = self.in_layer(x)

    ## our logo-llm module
    if self.is_gpt:
      outputs = self.gpt2(inputs_embeds=x_embed).hidden_states

      if self.version == 'first_last':
        low_feature, high_feature = outputs[5], outputs[-1]
      elif self.version == 'mean':
        low_feature = torch.mean(outputs[:self.num_layer], dim=-1, keepdim=True)
        high_feature = torch.mean(outputs[-self.num_layer:], dim=-1, keepdim=True)
      else:
        raise NotImplementedError
      outputs = self.mixer(x_embed, low_feature, high_feature)

    ## output projection layer
    outputs = self.out_layer(outputs.reshape(B * M, -1))
    outputs = rearrange(outputs, '(b m) l -> b l m', b=B)

    ## denormalize
    outputs = outputs * stdev
    outputs = outputs + means

    return outputs


class Mixer(nn.Module):
  def __init__(self, configs, device):
    super(Mixer, self).__init__()
    self.local_mixer = Mixer_Block(configs, device)
    self.global_mixer = Mixer_Block(configs, device)

  def forward(self, x, local_feature, global_feature):
    dec_local = self.local_mixer(x, local_feature)
    dec_global = self.global_mixer(x, global_feature)
    dec_out = (x + dec_local + dec_global) / 3
    return dec_out


class Mixer_Block(nn.Module):
  def __init__(self, configs, device):
    super(Mixer_Block, self).__init__()
    self.mlp = nn.Sequential(
      nn.Linear(configs.d_model * 2, configs.d_model),
      nn.ReLU(),
      nn.Linear(configs.d_model, configs.d_model)
    ).to(device)
    self.dropout = nn.Dropout(0.3).to(device)

  def forward(self, temporal_input, feature):
    input_concat = torch.cat([temporal_input, feature], dim=-1)
    output = self.dropout(self.mlp(input_concat))
    return output