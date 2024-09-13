```
llama_model_loader: loaded meta data with 33 key-value pairs and 292 tensors from /Users/theapemachine/.cache/huggingface/hub/models--lmstudio-community--Meta-Llama-3.1-8B-Instruct-GGUF/snapshots/8601e6db71269a2b12255ebdf09ab75becf22cc8/Meta-Llama-3.1-8B-Instruct-Q6_K.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = llama
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = Meta Llama 3.1 8B Instruct
llama_model_loader: - kv   3:                           general.finetune str              = Instruct
llama_model_loader: - kv   4:                           general.basename str              = Meta-Llama-3.1
llama_model_loader: - kv   5:                         general.size_label str              = 8B
llama_model_loader: - kv   6:                            general.license str              = llama3.1
llama_model_loader: - kv   7:                               general.tags arr[str,6]       = ["facebook", "meta", "pytorch", "llam...
llama_model_loader: - kv   8:                          general.languages arr[str,8]       = ["en", "de", "fr", "it", "pt", "hi", ...
llama_model_loader: - kv   9:                          llama.block_count u32              = 32
llama_model_loader: - kv  10:                       llama.context_length u32              = 131072
llama_model_loader: - kv  11:                     llama.embedding_length u32              = 4096
llama_model_loader: - kv  12:                  llama.feed_forward_length u32              = 14336
llama_model_loader: - kv  13:                 llama.attention.head_count u32              = 32
llama_model_loader: - kv  14:              llama.attention.head_count_kv u32              = 8
llama_model_loader: - kv  15:                       llama.rope.freq_base f32              = 500000.000000
llama_model_loader: - kv  16:     llama.attention.layer_norm_rms_epsilon f32              = 0.000010
llama_model_loader: - kv  17:                          general.file_type u32              = 18
llama_model_loader: - kv  18:                           llama.vocab_size u32              = 128256
llama_model_loader: - kv  19:                 llama.rope.dimension_count u32              = 128
llama_model_loader: - kv  20:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  21:                         tokenizer.ggml.pre str              = llama-bpe
llama_model_loader: - kv  22:                      tokenizer.ggml.tokens arr[str,128256]  = ["!", "\"", "#", "$", "%", "&", "'", ...
llama_model_loader: - kv  23:                  tokenizer.ggml.token_type arr[i32,128256]  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  24:                      tokenizer.ggml.merges arr[str,280147]  = ["Ġ Ġ", "Ġ ĠĠĠ", "ĠĠ ĠĠ", "...
llama_model_loader: - kv  25:                tokenizer.ggml.bos_token_id u32              = 128000
llama_model_loader: - kv  26:                tokenizer.ggml.eos_token_id u32              = 128009
llama_model_loader: - kv  27:                    tokenizer.chat_template str              = {{- bos_token }}\n{%- if custom_tools ...
llama_model_loader: - kv  28:               general.quantization_version u32              = 2
llama_model_loader: - kv  29:                      quantize.imatrix.file str              = /models_out/Meta-Llama-3.1-8B-Instruc...
llama_model_loader: - kv  30:                   quantize.imatrix.dataset str              = /training_dir/calibration_datav3.txt
llama_model_loader: - kv  31:             quantize.imatrix.entries_count i32              = 224
llama_model_loader: - kv  32:              quantize.imatrix.chunks_count i32              = 125
llama_model_loader: - type  f32:   66 tensors
llama_model_loader: - type q6_K:  226 tensors
llm_load_vocab: special tokens cache size = 256
llm_load_vocab: token to piece cache size = 0.7999 MB
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = llama
llm_load_print_meta: vocab type       = BPE
llm_load_print_meta: n_vocab          = 128256
llm_load_print_meta: n_merges         = 280147
llm_load_print_meta: vocab_only       = 0
llm_load_print_meta: n_ctx_train      = 131072
llm_load_print_meta: n_embd           = 4096
llm_load_print_meta: n_layer          = 32
llm_load_print_meta: n_head           = 32
llm_load_print_meta: n_head_kv        = 8
llm_load_print_meta: n_rot            = 128
llm_load_print_meta: n_swa            = 0
llm_load_print_meta: n_embd_head_k    = 128
llm_load_print_meta: n_embd_head_v    = 128
llm_load_print_meta: n_gqa            = 4
llm_load_print_meta: n_embd_k_gqa     = 1024
llm_load_print_meta: n_embd_v_gqa     = 1024
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-05
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: f_logit_scale    = 0.0e+00
llm_load_print_meta: n_ff             = 14336
llm_load_print_meta: n_expert         = 0
llm_load_print_meta: n_expert_used    = 0
llm_load_print_meta: causal attn      = 1
llm_load_print_meta: pooling type     = 0
llm_load_print_meta: rope type        = 0
llm_load_print_meta: rope scaling     = linear
llm_load_print_meta: freq_base_train  = 500000.0
llm_load_print_meta: freq_scale_train = 1
llm_load_print_meta: n_ctx_orig_yarn  = 131072
llm_load_print_meta: rope_finetuned   = unknown
llm_load_print_meta: ssm_d_conv       = 0
llm_load_print_meta: ssm_d_inner      = 0
llm_load_print_meta: ssm_d_state      = 0
llm_load_print_meta: ssm_dt_rank      = 0
llm_load_print_meta: model type       = 8B
llm_load_print_meta: model ftype      = Q6_K
llm_load_print_meta: model params     = 8.03 B
llm_load_print_meta: model size       = 6.14 GiB (6.56 BPW)
llm_load_print_meta: general.name     = Meta Llama 3.1 8B Instruct
llm_load_print_meta: BOS token        = 128000 '<|begin_of_text|>'
llm_load_print_meta: EOS token        = 128009 '<|eot_id|>'
llm_load_print_meta: LF token         = 128 'Ä'
llm_load_print_meta: EOT token        = 128009 '<|eot_id|>'
llm_load_print_meta: max token length = 256
llm_load_tensors: ggml ctx size =    0.27 MiB
ggml_backend_metal_log_allocated_size: allocated buffer, size =  6282.98 MiB, ( 6283.06 / 10922.67)
llm_load_tensors: offloading 32 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 33/33 layers to GPU
llm_load_tensors:        CPU buffer size =   410.98 MiB
llm_load_tensors:      Metal buffer size =  6282.98 MiB
.........................................................................................
llama_new_context_with_model: n_ctx      = 2048
llama_new_context_with_model: n_batch    = 512
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: flash_attn = 0
llama_new_context_with_model: freq_base  = 500000.0
llama_new_context_with_model: freq_scale = 1
ggml_metal_init: allocating
ggml_metal_init: found device: Apple M3
ggml_metal_init: picking default device: Apple M3
ggml_metal_init: using embedded metal library
ggml_metal_init: GPU name:   Apple M3
ggml_metal_init: GPU family: MTLGPUFamilyApple9  (1009)
ggml_metal_init: GPU family: MTLGPUFamilyCommon3 (3003)
ggml_metal_init: GPU family: MTLGPUFamilyMetal3  (5001)
ggml_metal_init: simdgroup reduction support   = true
ggml_metal_init: simdgroup matrix mul. support = true
ggml_metal_init: hasUnifiedMemory              = true
ggml_metal_init: recommendedMaxWorkingSetSize  = 11453.25 MB
llama_kv_cache_init:      Metal KV buffer size =   256.00 MiB
llama_new_context_with_model: KV self size  =  256.00 MiB, K (f16):  128.00 MiB, V (f16):  128.00 MiB
llama_new_context_with_model:        CPU  output buffer size =     0.49 MiB
llama_new_context_with_model:      Metal compute buffer size =   258.50 MiB
llama_new_context_with_model:        CPU compute buffer size =    12.01 MiB
llama_new_context_with_model: graph nodes  = 1030
llama_new_context_with_model: graph splits = 2
AVX = 0 | AVX_VNNI = 0 | AVX2 = 0 | AVX512 = 0 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | AVX512_BF16 = 0 | FMA = 0 | NEON = 1 | SVE = 0 | ARM_FMA = 1 | F16C = 0 | FP16_VA = 1 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 0 | SSSE3 = 0 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 |
Model metadata: {'quantize.imatrix.file': '/models_out/Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct.imatrix', 'quantize.imatrix.chunks_count': '125', 'tokenizer.chat_template': '{{- bos_token }}\n{%- if custom_tools is defined %}\n    {%- set tools = custom_tools %}\n{%- endif %}\n{%- if not tools_in_user_message is defined %}\n    {%- set tools_in_user_message = true %}\n{%- endif %}\n{%- if not date_string is defined %}\n    {%- set date_string = "26 Jul 2024" %}\n{%- endif %}\n\n{#- This block extracts the system message, so we can slot it into the right place. #}\n{%- if messages[0][\'role\'] == \'system\' %}\n    {%- set system_message = messages[0][\'content\']|trim %}\n    {%- set messages = messages[1:] %}\n{%- else %}\n    {%- set system_message = "" %}\n{%- endif %}\n\n{#- System message + builtin tools #}\n{{- "<|start_header_id|>system<|end_header_id|>\\n\\n" }}\n{%- if builtin_tools is defined or tools is not none %}\n    {{- "Environment: ipython\\n" }}\n{%- endif %}\n{%- if builtin_tools is defined %}\n    {{- "Tools: " + builtin_tools | reject(\'equalto\', \'code_interpreter\') | join(", ") + "\\n\\n"}}\n{%- endif %}\n{{- "Cutting Knowledge Date: December 2023\\n" }}\n{{- "Today Date: " + date_string + "\\n\\n" }}\n{%- if tools is not none and not tools_in_user_message %}\n    {{- "You have access to the following functions. To call a function, please respond with JSON for a function call." }}\n    {{- \'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.\' }}\n    {{- "Do not use variables.\\n\\n" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- "\\n\\n" }}\n    {%- endfor %}\n{%- endif %}\n{{- system_message }}\n{{- "<|eot_id|>" }}\n\n{#- Custom tools are passed in a user message with some extra guidance #}\n{%- if tools_in_user_message and not tools is none %}\n    {#- Extract the first user message so we can plug it in here #}\n    {%- if messages | length != 0 %}\n        {%- set first_user_message = messages[0][\'content\']|trim %}\n        {%- set messages = messages[1:] %}\n    {%- else %}\n        {{- raise_exception("Cannot put tools in the first user message when there\'s no first user message!") }}\n{%- endif %}\n    {{- \'<|start_header_id|>user<|end_header_id|>\\n\\n\' -}}\n    {{- "Given the following functions, please respond with a JSON for a function call " }}\n    {{- "with its proper arguments that best answers the given prompt.\\n\\n" }}\n    {{- \'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.\' }}\n    {{- "Do not use variables.\\n\\n" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- "\\n\\n" }}\n    {%- endfor %}\n    {{- first_user_message + "<|eot_id|>"}}\n{%- endif %}\n\n{%- for message in messages %}\n    {%- if not (message.role == \'ipython\' or message.role == \'tool\' or \'tool_calls\' in message) %}\n        {{- \'<|start_header_id|>\' + message[\'role\'] + \'<|end_header_id|>\\n\\n\'+ message[\'content\'] | trim + \'<|eot_id|>\' }}\n    {%- elif \'tool_calls\' in message %}\n        {%- if not message.tool_calls|length == 1 %}\n            {{- raise_exception("This model only supports single tool-calls at once!") }}\n        {%- endif %}\n        {%- set tool_call = message.tool_calls[0].function %}\n        {%- if builtin_tools is defined and tool_call.name in builtin_tools %}\n            {{- \'<|start_header_id|>assistant<|end_header_id|>\\n\\n\' -}}\n            {{- "<|python_tag|>" + tool_call.name + ".call(" }}\n            {%- for arg_name, arg_val in tool_call.arguments | items %}\n                {{- arg_name + \'="\' + arg_val + \'"\' }}\n                {%- if not loop.last %}\n                    {{- ", " }}\n                {%- endif %}\n                {%- endfor %}\n            {{- ")" }}\n        {%- else  %}\n            {{- \'<|start_header_id|>assistant<|end_header_id|>\\n\\n\' -}}\n            {{- \'{"name": "\' + tool_call.name + \'", \' }}\n            {{- \'"parameters": \' }}\n            {{- tool_call.arguments | tojson }}\n            {{- "}" }}\n        {%- endif %}\n        {%- if builtin_tools is defined %}\n            {#- This means we\'re in ipython mode #}\n            {{- "<|eom_id|>" }}\n        {%- else %}\n            {{- "<|eot_id|>" }}\n        {%- endif %}\n    {%- elif message.role == "tool" or message.role == "ipython" %}\n        {{- "<|start_header_id|>ipython<|end_header_id|>\\n\\n" }}\n        {%- if message.content is mapping or message.content is iterable %}\n            {{- message.content | tojson }}\n        {%- else %}\n            {{- message.content }}\n        {%- endif %}\n        {{- "<|eot_id|>" }}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- \'<|start_header_id|>assistant<|end_header_id|>\\n\\n\' }}\n{%- endif %}\n', 'tokenizer.ggml.eos_token_id': '128009', 'general.type': 'model', 'tokenizer.ggml.bos_token_id': '128000', 'tokenizer.ggml.pre': 'llama-bpe', 'tokenizer.ggml.model': 'gpt2', 'llama.embedding_length': '4096', 'llama.vocab_size': '128256', 'llama.attention.head_count_kv': '8', 'general.finetune': 'Instruct', 'general.file_type': '18', 'llama.block_count': '32', 'general.size_label': '8B', 'llama.attention.layer_norm_rms_epsilon': '0.000010', 'llama.feed_forward_length': '14336', 'general.quantization_version': '2', 'llama.rope.dimension_count': '128', 'general.license': 'llama3.1', 'llama.attention.head_count': '32', 'quantize.imatrix.entries_count': '224', 'llama.context_length': '131072', 'general.architecture': 'llama', 'general.basename': 'Meta-Llama-3.1', 'llama.rope.freq_base': '500000.000000', 'quantize.imatrix.dataset': '/training_dir/calibration_datav3.txt', 'general.name': 'Meta Llama 3.1 8B Instruct'}
Available chat formats from metadata: chat_template.default
Using gguf chat template: {{- bos_token }}
{%- if custom_tools is defined %}
    {%- set tools = custom_tools %}
{%- endif %}
{%- if not tools_in_user_message is defined %}
    {%- set tools_in_user_message = true %}
{%- endif %}
{%- if not date_string is defined %}
    {%- set date_string = "26 Jul 2024" %}
{%- endif %}

{#- This block extracts the system message, so we can slot it into the right place. #}
{%- if messages[0]['role'] == 'system' %}
    {%- set system_message = messages[0]['content']|trim %}
    {%- set messages = messages[1:] %}
{%- else %}
    {%- set system_message = "" %}
{%- endif %}

{#- System message + builtin tools #}
{{- "<|start_header_id|>system<|end_header_id|>\n\n" }}
{%- if builtin_tools is defined or tools is not none %}
    {{- "Environment: ipython\n" }}
{%- endif %}
{%- if builtin_tools is defined %}
    {{- "Tools: " + builtin_tools | reject('equalto', 'code_interpreter') | join(", ") + "\n\n"}}
{%- endif %}
{{- "Cutting Knowledge Date: December 2023\n" }}
{{- "Today Date: " + date_string + "\n\n" }}
{%- if tools is not none and not tools_in_user_message %}
    {{- "You have access to the following functions. To call a function, please respond with JSON for a function call." }}
    {{- 'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.' }}
    {{- "Do not use variables.\n\n" }}
    {%- for t in tools %}
        {{- t | tojson(indent=4) }}
        {{- "\n\n" }}
    {%- endfor %}
{%- endif %}
{{- system_message }}
{{- "<|eot_id|>" }}

{#- Custom tools are passed in a user message with some extra guidance #}
{%- if tools_in_user_message and not tools is none %}
    {#- Extract the first user message so we can plug it in here #}
    {%- if messages | length != 0 %}
        {%- set first_user_message = messages[0]['content']|trim %}
        {%- set messages = messages[1:] %}
    {%- else %}
        {{- raise_exception("Cannot put tools in the first user message when there's no first user message!") }}
{%- endif %}
    {{- '<|start_header_id|>user<|end_header_id|>\n\n' -}}
    {{- "Given the following functions, please respond with a JSON for a function call " }}
    {{- "with its proper arguments that best answers the given prompt.\n\n" }}
    {{- 'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.' }}
    {{- "Do not use variables.\n\n" }}
    {%- for t in tools %}
        {{- t | tojson(indent=4) }}
        {{- "\n\n" }}
    {%- endfor %}
    {{- first_user_message + "<|eot_id|>"}}
{%- endif %}

{%- for message in messages %}
    {%- if not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}
        {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' }}
    {%- elif 'tool_calls' in message %}
        {%- if not message.tool_calls|length == 1 %}
            {{- raise_exception("This model only supports single tool-calls at once!") }}
        {%- endif %}
        {%- set tool_call = message.tool_calls[0].function %}
        {%- if builtin_tools is defined and tool_call.name in builtin_tools %}
            {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' -}}
            {{- "<|python_tag|>" + tool_call.name + ".call(" }}
            {%- for arg_name, arg_val in tool_call.arguments | items %}
                {{- arg_name + '="' + arg_val + '"' }}
                {%- if not loop.last %}
                    {{- ", " }}
                {%- endif %}
                {%- endfor %}
            {{- ")" }}
        {%- else  %}
            {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' -}}
            {{- '{"name": "' + tool_call.name + '", ' }}
            {{- '"parameters": ' }}
            {{- tool_call.arguments | tojson }}
            {{- "}" }}
        {%- endif %}
        {%- if builtin_tools is defined %}
            {#- This means we're in ipython mode #}
            {{- "<|eom_id|>" }}
        {%- else %}
            {{- "<|eot_id|>" }}
        {%- endif %}
    {%- elif message.role == "tool" or message.role == "ipython" %}
        {{- "<|start_header_id|>ipython<|end_header_id|>\n\n" }}
        {%- if message.content is mapping or message.content is iterable %}
            {{- message.content | tojson }}
        {%- else %}
            {{- message.content }}
        {%- endif %}
        {{- "<|eot_id|>" }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
{%- endif %}

Using chat eos_token: <|eot_id|>
Using chat bos_token: <|begin_of_text|>
```