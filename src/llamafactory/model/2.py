import random
import torch
from transformers import MarianMTModel, MarianTokenizer
from typing import Dict


# 加载翻译模型 (这里使用MarianMT模型作为示例)
def load_translation_model():
    # 使用英语到法语、法语到英语的双向翻译模型
    model_name = "Helsinki-NLP/opus-mt-en-fr"  # 英法翻译模型
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    return model, tokenizer


# 回译方法：将句子翻译成目标语言，再翻译回原语言
def back_translation(tokens: torch.Tensor, model, tokenizer, source_lang='en', target_lang='fr'):
    # 将tokenized输入转换回句子
    sentence = tokenizer.decode(tokens[0], skip_special_tokens=True)

    # 翻译句子到目标语言
    translated = model.generate(**tokenizer(sentence, return_tensors="pt", padding=True))
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)

    # 将翻译后的句子翻译回原语言
    reverse_model_name = f"Helsinki-NLP/opus-mt-{target_lang}-{source_lang}"
    reverse_model = MarianMTModel.from_pretrained(reverse_model_name)
    reverse_tokenizer = MarianTokenizer.from_pretrained(reverse_model_name)

    reverse_translated = reverse_model.generate(**reverse_tokenizer(translated_text, return_tensors="pt", padding=True))
    reverse_translated_text = reverse_tokenizer.decode(reverse_translated[0], skip_special_tokens=True)

    # 将回译后的文本再编码成token_ids
    return reverse_tokenizer.encode(reverse_translated_text, add_special_tokens=False)


# 同义词替换
def synonym_replacement(tokens: torch.Tensor, tokenizer, p: float = 0.1):
    new_tokens = tokens.clone()
    for i, token_id in enumerate(tokens[0]):
        if random.random() < p:
            word = tokenizer.decode([token_id])
            # 这里可以使用词库或简单的示例来做同义词替换
            synonyms = [word]  # 这里仅做演示，通常会查找同义词
            new_tokens[0, i] = tokenizer.encode(random.choice(synonyms), add_special_tokens=False)[0]
    return new_tokens


# 随机删除词
def random_deletion(tokens: torch.Tensor, p: float = 0.1):
    new_tokens = [token for token in tokens[0] if random.random() > p]
    return torch.tensor([new_tokens if len(new_tokens) > 0 else [random.choice(tokens[0])]])


# 随机插入词
def random_insertion(tokens: torch.Tensor, tokenizer, p: float = 0.1):
    new_tokens = tokens.clone()
    for _ in range(int(tokens.size(1) * p)):  # 随机插入操作
        new_tokens = torch.cat([new_tokens, torch.tensor([[random.choice(tokens[0])]])], dim=1)
    return new_tokens


# 随机交换词
def random_swap(tokens: torch.Tensor, p: float = 0.1):
    new_tokens = tokens.clone()
    for _ in range(int(tokens.size(1) * p)):
        idx1, idx2 = random.sample(range(tokens.size(1)), 2)
        new_tokens[0, idx1], new_tokens[0, idx2] = new_tokens[0, idx2], new_tokens[0, idx1]
    return new_tokens


# 主增强函数：包括回译和其他文本增强操作
def get_tta_transforms(inputs: Dict[str, torch.Tensor], n: int = 10, p: float = 0.1, model=None, tokenizer=None):
    input_ids = inputs['input_ids']
    mask = inputs['mask']
    
    batch_size = input_ids.size(0)
    augmented_inputs = []

    for _ in range(n):  # 进行n次增强
        random.seed()  # 保证每次增强时种子不同

        # 创建一个新的batch，用于存储增强后的结果
        batch_augmented = []

        for i in range(batch_size):  # 对每个句子进行增强
            tokens = input_ids[i:i+1]  # 取出一个句子

            # 随机选择增强操作
            operations = [synonym_replacement, random_deletion, random_insertion, random_swap]
            random.shuffle(operations)  # 随机打乱操作顺序

            for op in operations:
                if op == synonym_replacement:
                    tokens = synonym_replacement(tokens, tokenizer, p)
                elif op == random_deletion:
                    tokens = random_deletion(tokens, p)
                elif op == random_insertion:
                    tokens = random_insertion(tokens, tokenizer, p)
                elif op == random_swap:
                    tokens = random_swap(tokens, p)

            # 执行回译
            if random.random() < 0.5:  # 50%的概率进行回译
                tokens = back_translation(tokens, model, tokenizer)

            # 将增强后的句子保存
            batch_augmented.append(tokens)

        # 将增强后的batch加入结果中
        augmented_inputs.append({
            'input_ids': torch.cat(batch_augmented, dim=0),
            'mask': mask  # 保留原始mask，或根据需求修改mask
        })
    
    return augmented_inputs


# 示例输入
sentence = ["The capital of France is Paris.", "The capital of Italy is Rome."]
model, tokenizer = load_translation_model()

# 将输入句子转换为tokenized格式
inputs = {
    'input_ids': torch.tensor([tokenizer.encode(sentence[i], add_special_tokens=False) for i in range(len(sentence))]),
    'mask': torch.tensor([[1] * len(tokenizer.encode(sentence[i], add_special_tokens=False)) for i in range(len(sentence))])
}

# 获取10次增强的结果
augmented_inputs = get_tta_transforms(inputs, n=10, model=model, tokenizer=tokenizer)

# 打印增强后的结果
for idx, augmented_input in enumerate(augmented_inputs):
    print(f"Augmented Input {idx + 1}:")
    for i, tokens in enumerate(augmented_input['input_ids']):
        print(f"  Sentence {i + 1}: {tokenizer.decode(tokens, skip_special_tokens=True)}")
