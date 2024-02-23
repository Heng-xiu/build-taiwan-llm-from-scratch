# -*- coding: utf-8 -*-
"""
@author:Heng-Shiou Sheu(hengshiousheu@gmail.com)
@description: Build chinese tokenizer from corpus txt

# Modified from the following repor
- https://github.com/shibing624/MedicalGPT/blob/main/build_domain_tokenizer.py
- https://github.com/Heng-xiu/BPE-tokenizer-from-zh-wiki/blob/main/02_train_sp.py
# train sentencepiece model from `corpus.txt` and makes `m.model` and `m.vocab`
# `m.vocab` is just a reference. not used in the segmentation.
# spm.SentencePieceTrainer.train('--input=data/pretrain/tianlongbabu.txt --model_prefix=m --vocab_size=20000')

"""

import argparse
import sentencepiece as spm


def main():

    spm.SentencePieceTrainer.train(
        input='wiki.txt',
        model_prefix='zh_wiki_bpe_model_sp/zh_wiki_bpe_hf',
        user_defined_symbols=["，","。","：","？","（","）","「","」"],
        max_sentence_length=2048,
        split_digits=True,
        model_type="bpe",
        byte_fallback=True,
        train_extremely_large_corpus=True,
        vocab_size=50000,
        split_by_unicode_script=True,
        allow_whitespace_only_pieces=True,
        remove_extra_whitespaces=False,
        normalization_rule_name="nmt_nfkc_cf",
        input_sentence_size=12000000,
        shuffle_input_sentence=True
    )

    # makes segmenter instance and loads the model file (m.model)
    sp = spm.SentencePieceProcessor()
    
    # LlamaTokenizer can use spm model file directly
    chinese_sp_model_file = "zh_wiki_bpe_model_sp/zh_wiki_bpe.model"
    tokenizer = LlamaTokenizerFast(vocab_file=chinese_sp_model_file)
    tokenizer.save_pretrained("zh_wiki_bpe_model_hf")
    print("done")

    # encode: text => id
    print(sp.encode_as_pieces('潜伏性感染又称潜在性感染。慕容复来到河边,this is a test'))
    print(sp.encode_as_ids('this is a test'))

    # decode: id => text
    print(sp.decode_pieces(['▁This', '▁is', '▁a', '▁t', 'est']))
    # print(sp.decode_ids([209, 31, 9, 375, 586]))
    
    


if __name__ == '__main__':
    main()