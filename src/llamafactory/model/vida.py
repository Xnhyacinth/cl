from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit
from transformers import T5ForConditionalGeneration, LlamaForCausalLM
import PIL
import torchvision.transforms as transforms
from .inject_vida import *
from time import time
import logging


# , iteration):
def update_ema_variables(ema_model, model, alpha_teacher, alpha_vida):
    # for ema_param, param in zip(ema_model.parameters(), model.parameters()):
    #     ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    # return ema_model
    for ema_param, (name, param) in zip(ema_model.parameters(), model.named_parameters()):
        # ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
        if "vida_" in name:
            ema_param.data[:] = alpha_vida * ema_param[:].data[:] + \
                (1 - alpha_vida) * param[:].data[:]
        else:
            ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + \
                (1 - alpha_teacher) * param[:].data[:]
    return ema_model


class LlamaVida(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.wrap_model()

    def wrap_model(self):
        inject_trainable_vida(self, target_replace_module=[
                              "LlamaMLP", "LlamaSdpaAttention", "LlamaAttention", "LlamaFlashAttention2"], r=self.config.vida_rank1, r2=self.config.vida_rank2)

    def unwrap_model(self):
        uninject_trainable_vida(self, target_replace_module=[
                                "LlamaMLP", "LlamaSdpaAttention", "LlamaAttention", "LlamaFlashAttention2"])

    def load_model(self, state_dict):
        self.unwrap_model()
        self.load_state_dict(state_dict)
        self.wrap_model()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

    def generate(self, **kwargs):
        # kwargs.pop('labels', None)
        return super().generate(**kwargs)


class T5Vida(T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.wrap_model()

    def wrap_model(self):
        inject_trainable_vida(self, target_replace_module=[
                              "T5Attention", "T5DenseActDense"], r=self.config.vida_rank1, r2=self.config.vida_rank2)

    def unwrap_model(self):
        uninject_trainable_vida(self, target_replace_module=[
                                "T5Attention", "T5DenseActDense"])

    def load_model(self, state_dict):
        self.unwrap_model()
        self.load_state_dict(state_dict)
        self.wrap_model()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    def generate(self, **kwargs):
        return super().generate(**kwargs)




@torch.jit.script
def softmax_entropy(x, x_ema):  # -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -0.5*(x_ema.softmax(1) * x.log_softmax(1)).sum(1)-0.5*(x.softmax(1) * x_ema.log_softmax(1)).sum(1)


def collect_params(model):
    """Collect all trainable parameters.

    Walk the model's modules and collect all parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    vida_params_list = []
    model_params_lst = []
    for name, param in model.named_parameters():
        if 'vida_' in name:
            vida_params_list.append(param)
        else:
            model_params_lst.append(param)
    return model_params_lst, vida_params_list


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    model_anchor = deepcopy(model)
    optimizer_state = deepcopy(optimizer.state_dict())
    ema_model = deepcopy(model)
    for param in ema_model.parameters():
        param.detach_()
    return model_state, optimizer_state, ema_model, model_anchor


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model, cfg):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = model.cpu()
    vida_params, vida_names = inject_trainable_vida(model=model, target_replace_module=["T5Attention", "T5DenseActDense"],
                                                    r=cfg.vida_rank1, r2=cfg.vida_rank2)
    if cfg.vida_name_or_path != None:
        model = torch.nn.DataParallel(model)  # make parallel
        checkpoint = torch.load(f'{cfg.vida_name_or_path}/vida.pt')
        model.load_state_dict(checkpoint, strict=True)
    # if cfg.TEST.ckpt!=None:
    #     checkpoint = torch.load(cfg.TEST.ckpt)
    #     model.load_state_dict(checkpoint, strict=True)
    # model.to(device)
    # model.train()
    return model


def check_model(model):
    """Check model for compatability with tent."""
    is_training = model.training
    assert is_training, "tent needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "tent needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "tent should not update all params: " \
                               "check which require grad"
    has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
    assert has_bn, "tent needs normalization for its optimization"
