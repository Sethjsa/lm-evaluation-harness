#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17.12.21 09:54
# @Author  : Wen Lai
# @Site    : 
# @File    : remove_punctuation.py
# @Usage information: 

# Copyright (c) 2021-present, CIS, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


"""
Give no of correct language for a test document output
Also look at languages of fewshot prompts - split on ":" or "{target-lang}"

"""
import argparse
import os
from langcodes import standardize_tag
import fasttext
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from pprint import pprint
import json
from glob import glob

model_path = hf_hub_download(repo_id="facebook/fasttext-language-identification", filename="model.bin")
model = fasttext.load_model(model_path)


modifiers = [
    # "xglm-base",
    # "xglm-base-trim",
    # "verbose-base",
    # "verbose-base-trim",
    # "xglm-label",
    # "xglm-label-trim",
    # "xglm-rand-label",
    # "xglm-rand-label-trim",
    # "xglm-keywords10-seen200",
    # "xglm-keywords10-seen200-trim",
    # "xglm-rand-keywords10-seen200",
    # "xglm-rand-keywords10-seen200-trim",
    # "xglm-keywords10-all500",
    # "xglm-keywords10-all500-trim",
    # "xglm-keywords30-seen200",
    # "xglm-keywords30-seen200-trim",
    # "xglm-topic3shot-seen200",
    # "xglm-topic3shot-seen200-trim",
    # "xglm-topic3shot-all500",
    # "xglm-topic3shot-all500-trim",
    # "xglm-topic3shot-cc100",
    # "xglm-topic3shot-cc100-trim",
    # "xglm-topic1shot-seen200",
    # "xglm-topic1shot-seen200-trim",
    # "xglm-rand-topic3shot-seen200",
    # "xglm-rand-topic3shot-seen200-trim",
    # "xglm-rand-domain3shot-langs",
    # "xglm-rand-domain3shot-langs-trim",
    # "xglm-rand-domain3shot-all",
    # "xglm-rand-domain3shot-all-trim",
    # "xglm-truerand-3shot-langs",
    # "xglm-truerand-3shot-langs-trim",
    # "xglm-truerand-3shot-all",
    # "xglm-truerand-3shot-all-trim",
    "xglm-topic5shot-seen200",
    "xglm-topic5shot-seen200-trim",
    "xglm-keywords10-cc100",
    "xglm-keywords10-cc100-trim",
    "xglm-keywords10-topic3shot-seen200",
    "xglm-keywords10-topic3shot-seen200-trim",
    ]

def main(args):
    # domain_list = ["OpenSubtitles"]
    # pre_list = ["en-fr", "de-en", "cs-en", "en-fi"]
    ### domain_list
    domain_list = ['EMEA', 'Tanzil', 'JRC-Acquis', 'KDE4', 'QED', 'TED2020', 'CCAligned', 'OpenSubtitles']
    # domain_list = ['EMEA', 'JRC-Acquis', 'KDE4', 'QED', 'TED2020', 'CCAligned', 'OpenSubtitles']
    # domain_list = ["EMEA"]
    # lang_pairs = ["en-fr"]
    ### pre_list
    lang_pairs = ["en-cs", "en-de", "en-fi", "en-fr", "en-lt", "en-ro", "en-ta","cs-en", "fr-en", "lt-en", "ro-en"]
    # lang_pairs = ["cs-en", "fr-en", "lt-en", "ro-en"]


    # print('Processing file {}'.format(args.input))

    for modifier in modifiers:
        for lang_pair in lang_pairs:
            for domain in domain_list:
                base_path = "/scratch/saycock/topic/lm-evaluation-harness/outputs/"
                base_path += f"task={domain}-{lang_pair}."
                # base_path += f"modifier={args.modifier}."
                base_path += f"modifier={modifier}"
                base_path += "*outputs.json"

                files = sorted(glob(base_path))

                langs = lang_pair.split("-")
                langs = [standardize_tag(lang) for lang in langs]

                # for filename in glob(base_path):
                try:
                    if len(files) > 1:
                        filename = files[-1]
                    else:
                        filename = files[0]

                    with open(filename, "r") as json_file:
                        # splut = filename.split(".")
                        # splut[1] = splut[1] + "-trim"
                        # splut = ".".join(splut)
                        # print(splut)
                        data = json.load(json_file)

                    with open(filename, "w") as output_file:
                                
                        # data = json.load(json_file)

                        for example in data:
                            pred = standardize_tag(model.predict(example["logit_0"])[0][0].split("__")[2], macro=True)
                            target = standardize_tag(model.predict(example["target"])[0][0].split("__")[2], macro=True)
                            if pred == target:
                                example["correct"] = 1
                            else:
                                example["correct"] = 0

                        json.dump(data, output_file, indent=4, ensure_ascii=False)
                    
                    results = filename.replace("_outputs.json", "_results.json")

                    with open(results, "r") as file1:
                        data1 = json.load(file1)
                        total = 0
                        for doc1 in data:
                            total += doc1["correct"]
                        langids = total                

                    with open(results, "w") as results_file:
                        task = results.split(".")[0].split("=")[1]
                        print(f"{task}-{modifier}")
                        try:
                            data1[task]["langids"] = langids
                            json.dump(data1, results_file, indent=4, ensure_ascii=False)
                        except KeyError:
                            pass
            
                except (FileNotFoundError, TypeError, IndexError, json.decoder.JSONDecodeError):
                        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--modifier", required=False, type=str, help='input path (merge_file)')
    parser.add_argument("--output", required=False, type=str, help='output path')
    parser.add_argument('--encoding', required=False, default='utf-8', help='character encoding for input/output')
    main(parser.parse_args())