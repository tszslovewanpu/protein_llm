import torch
from typing import Literal, Optional, Tuple
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from transformers.utils import add_start_docstrings_to_model_forward, replace_return_docstrings
from transformers.modeling_outputs import CausalLMOutputWithPast
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import LlamaForCausalLM
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import LlamaModel
from transformers.utils import add_start_docstrings_to_model_forward
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.cache_utils import Cache, DynamicCache, StaticCache
import json

LLAMA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance;
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
"""
# from transformers.models.llama.modeling_llama import LlamaModel
# from transformers.utils import add_start_docstrings_to_model_forward
# from transformers.modeling_outputs import BaseModelOutputWithPast
# from transformers.cache_utils import Cache, DynamicCache, StaticCache
# import json

# class LlamaModelProtein(LlamaModel):
#     def __init__(self, config):
#         super().__init__(config)
#         self.linear_map = nn.Linear(32, 4096)
#         # 自定义初始化 linear_map
#         nn.init.xavier_uniform_(self.linear_map.weight)
#         nn.init.zeros_(self.linear_map.bias)
#         self.query_table = self.load_query_table("/data/fcl/fcl/workspace/2024_35/240724_protein/16_read_datasets/codebook_embedding.json") ### torch.Size([512, 32]) 只保留了四位小数

#     def load_query_table(self, query_file):
#         with open(query_file, 'r') as f:
#             query_data = json.load(f)
#         return torch.tensor([query_data[str(i)] for i in range(1, 513)], dtype=torch.float32)

#     @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
#     def forward(
#         self,
#         input_ids: torch.LongTensor = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         past_key_values: Optional[List[torch.FloatTensor]] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#         cache_position: Optional[torch.LongTensor] = None,
#     ) -> Union[Tuple, BaseModelOutputWithPast]:
#         output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#         output_hidden_states = (
#             output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#         )
#         use_cache = use_cache if use_cache is not None else self.config.use_cache
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         if (input_ids is None) ^ (inputs_embeds is not None):
#             raise ValueError(
#                 "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
#             )

#         if self.gradient_checkpointing and self.training and use_cache:
#             logger.warning_once(
#                 "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
#             )
#             use_cache = False
#         # if inputs_embeds is None: ### 原来的代码
#             # # inputs_embeds = self.embed_tokens(input_ids) ### 原来的代码
#         if inputs_embeds is None:
#             batch_size, seq_len = input_ids.size()
#             # inputs_embeds = torch.zeros(batch_size, seq_len, 4096).to(input_ids.device)
#             inputs_embeds = torch.zeros(batch_size, seq_len, 4096, dtype=torch.bfloat16).to(input_ids.device)
#             mask = (input_ids >= 128256) & (input_ids <= 128767)
#             if mask.any():
#                 input_ids_masked = input_ids[mask]
#                 vec_32 = self.query_table[input_ids_masked - 128256]
#                 vec_4096 = self.linear_map(vec_32).to(inputs_embeds.dtype)
#                 inputs_embeds[mask] = vec_4096
#             inputs_embeds[~mask] = self.embed_tokens(input_ids[~mask]).to(inputs_embeds.dtype) # torch.Size([1, 1024, 4096])

# torch dtype 和原来代码不一样了！
# inputs_embeds
# tensor([[[-0.0034, -0.0033,  0.0015,  ...,  0.0048, -0.0085,  0.0071],
#          [-0.0099,  0.0024, -0.0021,  ...,  0.0082, -0.0040, -0.0035],
#          [-0.0078,  0.0046, -0.0040,  ...,  0.0046, -0.0018, -0.0071],
#          ...,
#          [ 0.0036, -0.0025, -0.0006,  ...,  0.0059,  0.0018,  0.0020],
#          [-0.0386, -0.0105,  0.0184,  ...,  0.0024, -0.0095,  0.0223],
#          [-0.0481, -0.0152,  0.0052,  ...,  0.0115,  0.0003,  0.0198]]],
#        device='cuda:2', grad_fn=<IndexPutBackward0>)
# inputs_embeds.dtype
# torch.float32

# 找到没有被mask的部分（延用原来embedding的部分）
# print(torch.nonzero(~mask, as_tuple=True))
# (tensor([0], device='cuda:3'), tensor([716], device='cuda:3'))


class LlamaModelProtein(LlamaModel):
    def __init__(self, config):
        super().__init__(config)
        self.linear_map = nn.Linear(32, 4096)
        nn.init.xavier_uniform_(self.linear_map.weight)
        nn.init.zeros_(self.linear_map.bias)
        self.query_table = self.load_query_table("/data/fcl/fcl/workspace/2024_35/240724_protein/16_read_datasets/codebook_embedding.json")

    # def load_query_table(self, query_file):
    #     with open(query_file, 'r') as f:
    #         query_data = json.load(f)
    #     return torch.tensor([query_data[str(i)] for i in range(1, 513)], dtype=torch.float32)
    def load_query_table(self, query_file):
        with open(query_file, 'r') as f:
            query_data = json.load(f)
        return torch.tensor([query_data[str(i)] for i in range(1, 513)], dtype=torch.float64)

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one")

        if self.gradient_checkpointing and self.training and use_cache:
            # logger.warning_once("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.")
            use_cache = False

        # if inputs_embeds is None:
        #     batch_size, seq_len = input_ids.size()
        #     inputs_embeds = torch.zeros(batch_size, seq_len, 4096, dtype=torch.bfloat16).to(input_ids.device)
        #     mask = (input_ids >= 128256) & (input_ids <= 128767)
        #     inputs_embeds[mask] = self.linear_map(self.query_table[input_ids[mask] - 128256])
        #     inputs_embeds[~mask] = self.embed_tokens(input_ids[~mask]) # torch.Size([1, 1024, 4096])
# inputs_embeds
# tensor([[[-0.0081, -0.0008,  0.0020,  ...,  0.0044, -0.0093, -0.0055],
#          [ 0.0007, -0.0020, -0.0045,  ..., -0.0123, -0.0019, -0.0055],
#          [-0.0081, -0.0008,  0.0020,  ...,  0.0044, -0.0093, -0.0055],
#          ...,
#          [-0.0420, -0.0103,  0.0322,  ...,  0.0068, -0.0046,  0.0248],
#          [-0.0265, -0.0013,  0.0176,  ...,  0.0020,  0.0108,  0.0164],
#          [-0.0339, -0.0105,  0.0300,  ...,  0.0062, -0.0052,  0.0160]]],
#        device='cuda:1', dtype=torch.bfloat16, grad_fn=<IndexPutBackward0>)
        if inputs_embeds is None:
            batch_size, seq_len = input_ids.size()
            inputs_embeds = torch.zeros(batch_size, seq_len, 4096, dtype=torch.bfloat16).to(input_ids.device)
            mask = (input_ids >= 128256) & (input_ids <= 128767)
            if mask.any():
                input_ids_masked = input_ids[mask]
                vec_32 = self.query_table[input_ids_masked - 128256].to(torch.bfloat16)
                vec_4096 = self.linear_map(vec_32)
                inputs_embeds[mask] = vec_4096
            inputs_embeds[~mask] = self.embed_tokens(input_ids[~mask]) # Embedding(128768, 4096) / print(self.linear_map.weight.dtype) torch.bfloat16
# inputs_embeds
# tensor([[[-0.0034, -0.0033,  0.0015,  ...,  0.0048, -0.0085,  0.0071],
#          [-0.0099,  0.0024, -0.0021,  ...,  0.0082, -0.0040, -0.0035],
#          [-0.0078,  0.0046, -0.0040,  ...,  0.0046, -0.0018, -0.0071],
#          ...,
#          [ 0.0036, -0.0025, -0.0006,  ...,  0.0059,  0.0018,  0.0020],
#          [-0.0386, -0.0105,  0.0184,  ...,  0.0024, -0.0095,  0.0223],
#          [-0.0481, -0.0152,  0.0052,  ...,  0.0115,  0.0003,  0.0198]]],
#        device='cuda:2', dtype=torch.bfloat16, grad_fn=<IndexPutBackward0>)

        return_legacy_cache = False
        if (
            use_cache and not isinstance(past_key_values, Cache) and not self.training
        ):  # kept for BC (non `Cache` `past_key_values` inputs)
            return_legacy_cache = True
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            # logger.warning_once(
            #     "We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. "
            #     "Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)"
            # )

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )
        hidden_states = inputs_embeds ### torch.Size([1, 1024, 4096])

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids) ### position_embeddings[0].shape = torch.Size([1, 1024, 128]) / position_embeddings[1].shape = torch.Size([1, 1024, 128]) 

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )



# from transformers.utils import add_start_docstrings_to_model_forward, replace_return_docstrings
# from transformers.modeling_outputs import CausalLMOutputWithPast
# from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
# from transformers import LlamaForCausalLM
# import torch.nn as nn

LLAMA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance;
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
"""
_CONFIG_FOR_DOC = "LlamaConfig"
class LlamaForCausalLMProtein(LlamaForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]
    # def __init__(self, config, train_batch_size, Protein_log_file_path):
    # def __init__(self, config, train_batch_size):
    def __init__(self, config):
        ######
        # super().__init__(config) # 不要直接用，因为父类LlamaForCausalLM(LlamaPreTrainedModel)的init里会创建一个LlamaModel的实例，这个实例我不希望他是由LlamaModel创建的，而是用我自定义（继承自LlamaModel）的子类创建
        ######

        ######
        # 这是 待继承父类class LlamaForCausalLM(LlamaPreTrainedModel):的init函数
                    # class LlamaForCausalLM(LlamaPreTrainedModel):
                    #     _tied_weights_keys = ["lm_head.weight"]

                    #     def __init__(self, config):
                    #         super().__init__(config)
                    #         self.model = LlamaModel(config) ### 居然不先来这里
                    #         self.vocab_size = config.vocab_size
                    #         self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

                    #         # Initialize weights and apply final processing
                    #         self.post_init()
        ######

        ######
        # 我将手动写下来父类class LlamaForCausalLM(LlamaPreTrainedModel)的init函数里的内容
        # 首先是爷爷类
        from transformers.models.llama.modeling_llama import LlamaPreTrainedModel # 先导入爷爷类
        LlamaPreTrainedModel.__init__(self, config) # 因此，手动初始化父类的父类
        # 最关键的
        self.model = LlamaModelProtein(config) ### 最关键的，一切的起因，是要这一行重写
        # 剩下的不变
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()
        ######

        ######
        # 下面写我自己class LlamaForCausalLMProtein(LlamaForCausalLM):的init函数里需要添加的东西
        # self.train_batch_size = train_batch_size
        # self._setup_Protein_logger(Protein_log_file_path)
        ######

    def forward( ### step1 先来到了这里进行forward的配置
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        # numbers: Optional[torch.Tensor] = None, ### biubiubiu
        # numbers_start_idx: Optional[torch.Tensor] = None, ### biubiubiu
        # numbers_end_idx: Optional[torch.Tensor] = None, ### biubiubiu
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions ### None 然后是 后面self的值，False
        output_hidden_states = ( ### None 然后是，最终选了这个self.config.output_hidden_states=False
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict ### None，然后是，最终选了这个self.config.use_return_dict=True
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model( ### 进入       ******   self.model = LlamaModelProtein(config) ### 最关键的，一切的起因，是要这一行重写
            # numbers=numbers, ### biubiubiu
            # numbers_start_idx=numbers_start_idx, ### biubiubiu
            # numbers_end_idx=numbers_end_idx, ### biubiubiu
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0] ### here torch.Size([3, 184, 4096]) ### one forward hidden_states.shape = torch.Size([21, 212, 4096]) 
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states) ### here torch.Size([3, 184, 128256])    self.lm_head = Linear(in_features=4096, out_features=128256, bias=False) ### logits.shape = torch.Size([21, 212, 128256]) one forward
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )