import random
from transformers import AutoTokenizer
from nltk.corpus import wordnet
import nltk
import random
import torch
nltk.download("wordnet")
from transformers import MarianMTModel, MarianTokenizer
import random
from transformers import AutoTokenizer
from nltk.corpus import wordnet
import nltk

nltk.download("wordnet")

def get_text_augmentation_transforms_for_ids(
    tokenizer,
    synonym_prob: float = 0.2,
    delete_prob: float = 0.1,
    insert_prob: float = 0.1,
    swap_prob: float = 0.1,
    pad_token_id: int = 0,
):
    """
    Returns a function to perform text augmentation on tokenized IDs.
    """

    # Back-translation helper
    def translate_back(sentence, src_lang="en", tgt_lang="fr"):
        model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        
        # Translate to target language
        translated = model.generate(**tokenizer(sentence, return_tensors="pt", padding=True))
        tgt_sentence = tokenizer.batch_decode(translated, skip_special_tokens=True)
        
        # Translate back to source language
        back_model_name = f"Helsinki-NLP/opus-mt-{tgt_lang}-{src_lang}"
        back_tokenizer = MarianTokenizer.from_pretrained(back_model_name)
        back_model = MarianMTModel.from_pretrained(back_model_name)
        
        back_translated = back_model.generate(**back_tokenizer(tgt_sentence, return_tensors="pt", padding=True))
        return back_tokenizer.batch_decode(back_translated, skip_special_tokens=True)
    
    # Synonym replacement helper
    def synonym_replacement(tokenized_ids, prob=0.2):
        tokens = tokenizer.convert_ids_to_tokens(tokenized_ids)
        augmented_tokens = []
        for token in tokens:
            if random.random() < prob and token not in tokenizer.special_tokens_map.values():
                synonyms = wordnet.synsets(token)
                if synonyms:
                    synonym = random.choice(synonyms).lemmas()[0].name().replace("_", " ")
                    augmented_tokens.append(synonym)
                else:
                    augmented_tokens.append(token)
            else:
                augmented_tokens.append(token)
        return tokenizer.convert_tokens_to_ids(augmented_tokens)

    # Masking random words helper
    def mask_words(tokenized_ids, prob=0.1):
        return [
            mask_token_id if random.random() < prob and token_id not in tokenizer.all_special_ids else token_id
            for token_id in tokenized_ids
        ]

    # Random deletion helper
    def random_deletion(tokenized_ids, prob=0.1):
        # Delete tokens with a certain probability
        return [
            token_id
            for token_id in tokenized_ids
            if random.random() > prob or token_id in tokenizer.all_special_ids
        ]

    # Random insertion helper
    def random_insertion(tokenized_ids, prob=0.1):
        # Insert random tokens into the sentence with a certain probability
        tokens = tokenizer.convert_ids_to_tokens(tokenized_ids)
        augmented_tokens = tokens[:]
        for i in range(len(tokens)):
            if random.random() < prob:
                random_token = random.choice(list(tokenizer.vocab.keys()))  # Random word from vocab
                augmented_tokens.insert(i, random_token)
        return tokenizer.convert_tokens_to_ids(augmented_tokens)

    # Random swap helper
    def random_swap(tokenized_ids, prob=0.1):
        tokens = tokenizer.convert_ids_to_tokens(tokenized_ids)
        augmented_tokens = tokens[:]
        for i in range(len(tokens)):
            if random.random() < prob:
                j = random.randint(0, len(tokens) - 1)
                augmented_tokens[i], augmented_tokens[j] = augmented_tokens[j], augmented_tokens[i]
        return tokenizer.convert_tokens_to_ids(augmented_tokens)

    def augment(inputs, tokenizer):
        """
        Apply a series of augmentations to a list of token IDs.
        """
        input_ids = inputs['input_ids']
        batch_size = input_ids.size(0)
        
        res = []
        for i in range(batch_size):
            ids =  input_ids[i]
            non_pad_ids = [token_id for token_id in ids if token_id != pad_token_id]
            
            operations = [synonym_replacement, random_deletion, random_insertion, random_swap]
            random.shuffle(operations)  # 随机打乱操作顺序

            for op in operations:
                if op == synonym_replacement:
                    augmented_ids = synonym_replacement(non_pad_ids, prob=synonym_prob)
                elif op == random_deletion:
                    augmented_ids = random_deletion(augmented_ids, prob=delete_prob)
                elif op == random_insertion:
                    augmented_ids = random_insertion(augmented_ids, prob=insert_prob)
                elif op == random_swap:
                    augmented_ids = random_swap(augmented_ids, prob=swap_prob)

            if random.random() < 0.5: 
                sentence = tokenizer.decode(augmented_ids, skip_special_tokens=True)
                lans = ["fr", "de"]
                random.shuffle(lans)
                sentence = translate_back(sentence, src_lang="en", tgt_lang=lans[0])
                # print(sentence)
                sentence = translate_back(sentence, src_lang="en", tgt_lang=lans[1])
                # print(sentence)
                augmented_ids = tokenizer(sentence, add_special_tokens=False, return_tensors="pt", padding=True)['input_ids'][0].tolist()
            # Pad back to the original length
            pad_length = len(ids) - len(augmented_ids)
            augmented_ids.extend([pad_token_id] * pad_length)

            res.append(augmented_ids)
        inputs['input_ids'] = torch.tensor(res)
        return inputs

    return augment


from transformers import AutoTokenizer

# 初始化 tokenizer
tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-large")

# 输入 tokenized_ids
sentence = ["The capital of France is Paris.", "The capital of Italy is Rome. Thank you very much for pointing out the issue."]
tokenized_ids = tokenizer(sentence, return_tensors="pt", padding=True)

# 获取增强函数
augment = get_text_augmentation_transforms_for_ids(
    tokenizer=tokenizer,
    synonym_prob=0.3,
    delete_prob=0.1,
    insert_prob=0.1,
    swap_prob=0.1,
    pad_token_id=tokenizer.pad_token_id,
    mask_token_id=tokenizer.mask_token_id,
)

for i in range(10):
    # 应用增强
    random.seed()
    augmented_ids = augment(tokenized_ids)
    breakpoint()
    # 转回增强后的句子
    augmented_sentence = tokenizer.batch_decode(augmented_ids['input_ids'], skip_special_tokens=False)

    print("Original Sentence:", sentence)
    # print("Tokenized IDs:", tokenized_ids)
    # print("Augmented Tokenized IDs:", augmented_ids)
    print("Augmented Sentence:", augmented_sentence)
