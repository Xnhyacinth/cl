
###
 # Copyright (c) 2024 by Huanxuan Liao, huanxuanliao@gmail.com, All Rights Reserved. 
 # @Author: Xnhyacinth, Xnhyacinth@qq.com
 # @Date: 2024-06-25 06:24:31
### 


num_gpus=${1:-"1"}
echo "GPU counts: ${num_gpus}"
gpus=${2:-"8"}
echo "GPU: ${gpus}"
model=${3:-"llama3-8b"}
lang=${4:-"en"}
n_shot=${5:-"0"}
template=${6:-"fewshot"}
split=${7:-"test"}
task=${8:-"bbh"}
temperature=${9:-"0.6"}
merge_path=${10:-"0"}
eval_path=${11:-"0"}
index_path=${12:-"0"}
retriever=${13:-"0"}
num_knowledge=${14:-"1"}
retrieval=${15:-"0"}
threshold=${16:-"68"}
extra_args=""

model_name_or_path=${model}
model="${model_name_or_path##*/}"
if [ "$model" = "llama3-8b" ];then
    model_name_or_path=meta-llama/Meta-Llama-3-8B
fi
if [ "$model" = "llama3-8b-inst" ];then
    model_name_or_path=meta-llama/Meta-Llama-3-8B-Instruct
    template=llama3
fi
if [ "$model" = "llama2-7b" ];then
    model_name_or_path=meta-llama/Llama-2-7b-hf
fi
if [ "$model" = "llama2-7b-chat" ];then
    model_name_or_path=meta-llama/Llama-2-7b-chat-hf
    template=llama2
fi
if [ "$model" = "tinyllama" ];then
    model_name_or_path=TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T
fi
if [ "$model" = "tinyllama-chat" ];then
    model_name_or_path=TinyLlama/TinyLlama-1.1B-Chat-v1.0
fi
if [ "$model" = "llama3-70b-inst" ];then
    model_name_or_path=meta-llama/Meta-Llama-3-70B-Instruct
    template=llama3
fi

split_task=$split

echo "split_task: ${split_task}"
vllm=False
save_path=saves/${model}/eval/${task}/${lang}_${split}_${n_shot}shot_${temperature}


if [[ $lang == *long* ]]
then
    vllm=True
    save_path=saves/${model}/eval/${task}/vllm_${lang}_${split}_${n_shot}shot_${temperature}
fi

if [ "$merge_path" != "0" ];then
    model_name_or_path=$merge_path
fi
if [ "$eval_path" != "0" ];then
    save_path="${eval_path}/${task}/${lang}_${split}_${n_shot}shot_${temperature}"
    if [[ $lang == *gen* ]]
    then
        save_path="${eval_path}/${task}/vllm_${lang}_${split}_${n_shot}shot_${temperature}"
    fi
fi
task="long/${task}"

# if [[ $task == *merge* ]];then
#     task="data/${task}"
# fi

echo "model_name_or_path: ${model_name_or_path}"
echo "template: ${template}"
echo "${n_shot}shot"
echo "save_path: ${save_path}"
echo "task: ${task}"
echo "lang: ${lang}"
echo ""

CUDA_VISIBLE_DEVICES=${gpus} llamafactory-cli eval \
    --model_name_or_path ${model_name_or_path} \
    --flash_attn fa2 \
    --task ${task} \
    --split ${split_task} \
    --template ${template} \
    --lang ${lang} \
    --n_shot ${n_shot} \
    --save_dir ${save_path} \
    --batch_size 32 \
    --vllm ${vllm} \
    --max_new_tokens 4096 \
    --do_sample False \
    --max_samples 100000 \
    --temperature ${temperature} \
    --top_p 0.9 \
    --top_k 50 \
    --repetition_penalty 1.0 \
    ${extra_args}