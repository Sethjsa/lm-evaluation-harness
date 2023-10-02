import csv
import argparse
from glob import glob
import json


def main(args):

    langs = args.langs
    lang_pairs = ["en-cs", "en-de", "en-fi", "en-fr", "en-lt", "en-ro", "en-ta", "cs-en", "fr-en", "lt-en", "ro-en"]
    # lang_pairs = ["cs-en", "fr-en", "lt-en", "ro-en"]
    all_domains = ["EMEA","JRC-Acquis","KDE4","OpenSubtitles","QED","Tanzil","TED2020","CCAligned"]
    
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
        "xglm-rand-keywords10-seen200",
        "xglm-rand-keywords10-seen200-trim",
        # "xglm-keywords10-all500",
        # "xglm-keywords10-all500-trim",
        # "xglm-topic3shot-seen200",
        # "xglm-topic3shot-seen200-trim",
        # "xglm-topic3shot-all500",
        # "xglm-topic3shot-all500-trim",
        # "xglm-topic3shot-cc100",
        # "xglm-topic3shot-cc100-trim",
        # "xglm-topic1shot-seen200",
        # "xglm-topic1shot-seen200-trim",
        "xglm-rand-topic3shot-seen200",
        "xglm-rand-topic3shot-seen200-trim"
        ]
    for modifier in modifiers:
        print(f"{modifier}:")

        print("BLEU:")
        with open(f"/scratch/saycock/topic/lm-evaluation-harness/results/alldomains_{modifier}_bleu_results.csv", "w", newline="") as csvfile:
            writer = csv.writer(csvfile, delimiter="\t")
            first_row = ["lang-pair"]
            first_row.extend(all_domains)
            writer.writerow(first_row)
            print(first_row)
            for lang_pair in lang_pairs:
                row = [f"{lang_pair}"]
                for domain in all_domains:
                    base_path = "/home/saycock/scratch/topic/lm-evaluation-harness/outputs/"
                    base_path += f"task={domain}-{lang_pair}."
                    base_path += f"modifier={modifier}."
                    base_path += "*results.json"
                    found = 0
                    files = sorted(glob(base_path))
                    # print(files)
                    # for filename in glob(base_path):
                    try:
                        if len(files) > 1:
                            filename = files[-1]
                        else:
                            filename = files[0]
                        with open(filename, "r") as json_file:
                            data = json.load(json_file)
                            bleu = round(data[f"{domain}-{lang_pair}"]["bleu"], 6)
                            row.append(bleu)
                            found = 1
                    except (KeyError, IndexError, json.decoder.JSONDecodeError):
                        pass
                    if not found:
                        row.append("NA")
                writer.writerow(row)
                print(row)
        
        print("COMET:")
            
        with open(f"/scratch/saycock/topic/lm-evaluation-harness/results/alldomains_{modifier}_comet_results.csv", "w", newline="") as csvfile:
            writer = csv.writer(csvfile, delimiter="\t")
            first_row = ["lang-pair"]
            first_row.extend(all_domains)
            writer.writerow(first_row)
            print(first_row)
            for lang_pair in lang_pairs:
                row = [f"{lang_pair}"]
                for domain in all_domains:
                    base_path = "/home/saycock/scratch/topic/lm-evaluation-harness/outputs/"
                    base_path += f"task={domain}-{lang_pair}."
                    base_path += f"modifier={modifier}."
                    base_path += "*results.json"
                    found = 0
                    files = sorted(glob(base_path))
                    # for filename in glob(base_path)
                    try:
                        if len(files) > 1:
                            filename = files[-1]
                        else:
                            filename = files[0]
                        with open(filename, "r") as json_file:
                            data = json.load(json_file)
                            comet = round(data[f"{domain}-{lang_pair}"]["comet"], 6)
                            row.append(comet)
                            found = 1
                    except (KeyError, IndexError, json.decoder.JSONDecodeError):
                        pass
                    if not found:
                        row.append("NA")
                            
                writer.writerow(row)
                print(row)

        with open(f"/scratch/saycock/topic/lm-evaluation-harness/results/alldomains_{modifier}_langid_results.csv", "w", newline="") as csvfile:
            writer = csv.writer(csvfile, delimiter="\t")
            first_row = ["lang-pair"]
            first_row.extend(all_domains)
            writer.writerow(first_row)
            print(first_row)
            for lang_pair in lang_pairs:
                row = [f"{lang_pair}"]
                for domain in all_domains:
                    base_path = "/home/saycock/scratch/topic/lm-evaluation-harness/outputs/"
                    base_path += f"task={domain}-{lang_pair}."
                    base_path += f"modifier={modifier}."
                    base_path += "*results.json"
                    found = 0
                    files = sorted(glob(base_path))
                    # for filename in glob(base_path)
                    try:
                        if len(files) > 1:
                            filename = files[-1]
                        else:
                            filename = files[0]
                        with open(filename, "r") as json_file:
                            data = json.load(json_file)
                            comet = data[f"{domain}-{lang_pair}"]["langids"]
                            row.append(comet)
                            found = 1
                    except (KeyError, IndexError, json.decoder.JSONDecodeError):
                        pass
                    if not found:
                        row.append("NA")
                            
                writer.writerow(row)
                print(row)


        with open(f"/scratch/saycock/topic/lm-evaluation-harness/results/alldomains_{modifier}_length_results.csv", "w", newline="") as csvfile:
            writer = csv.writer(csvfile, delimiter="\t")
            first_row = ["lang-pair"]
            first_row.extend(all_domains)
            writer.writerow(first_row)
            print(first_row)
            for lang_pair in lang_pairs:
                row = [f"{lang_pair}"]
                for domain in all_domains:
                    base_path = "/home/saycock/scratch/topic/lm-evaluation-harness/outputs/"
                    base_path += f"task={domain}-{lang_pair}."
                    base_path += f"modifier={modifier}."
                    base_path += "*results.json"
                    found = 0
                    files = sorted(glob(base_path))
                    # for filename in glob(base_path)
                    try:
                        if len(files) > 1:
                            filename = files[-1]
                        else:
                            filename = files[0]
                        with open(filename, "r") as json_file:
                            data = json.load(json_file)
                            comet = data[f"{domain}-{lang_pair}"]["av_len"]
                            row.append(comet)
                            found = 1
                    except (KeyError, IndexError, json.decoder.JSONDecodeError):
                        pass
                    if not found:
                        row.append("NA")
                            
                writer.writerow(row)
                print(row)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--langs", required=False, type=list, help='list of lang pairs')
    parser.add_argument("--domains", required=False, type=list, help='list of domains')
    parser.add_argument("--modifier", required=False, type=str, help='modifier')
    main(parser.parse_args())