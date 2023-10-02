import csv

modifiers = [
    "xglm-base",
    "xglm-base-trim",
    "verbose-base",
    "verbose-base-trim",
    "xglm-label",
    "xglm-label-trim",
    "xglm-rand-label",
    "xglm-rand-label-trim",
    "xglm-keywords10-seen200",
    "xglm-keywords10-seen200-trim",
    "xglm-rand-keywords10-seen200",
    "xglm-rand-keywords10-seen200-trim",
    "xglm-keywords10-all500",
    "xglm-keywords10-all500-trim",
    "xglm-topic3shot-seen200",
    "xglm-topic3shot-seen200-trim",
    "xglm-topic3shot-all500",
    "xglm-topic3shot-all500-trim",
    "xglm-topic3shot-cc100",
    "xglm-topic3shot-cc100-trim",
    "xglm-topic1shot-seen200",
    "xglm-topic1shot-seen200-trim",
    "xglm-rand-topic3shot-seen200",
    "xglm-rand-topic3shot-seen200-trim"
    ]

lang_pairs = ["en-cs", "en-de", "en-fi", "en-fr", "en-lt", "en-ro", "en-ta", "cs-en", "fr-en", "lt-en", "ro-en"]

all_domains = ["EMEA","JRC-Acquis","KDE4","OpenSubtitles","QED","Tanzil","TED2020","CCAligned"]
evals = ["bleu", "comet", "langid", "length"]
path = "/scratch/saycock/topic/lm-evaluation-harness/results/"
# {args.modifier}_comet_results.csv", "w", newline="")

for metric in evals:
    with open(f"{path}master_{metric}_results.csv", "w") as master:
        masterwriter = csv.writer(master, delimiter="\t")
        firstrow = ["modifier", "lang-pair", "EMEA","JRC-Acquis","KDE4","OpenSubtitles","QED","Tanzil","TED2020","CCAligned"]
        masterwriter.writerow(firstrow)
        for pair in lang_pairs:
            with open(f"{path}{pair}_{metric}_results.csv", "w") as outfile:
                writer = csv.writer(outfile, delimiter="\t")
                data = []
                firstrow = ["modifier", "lang-pair", "EMEA","JRC-Acquis","KDE4","OpenSubtitles","QED","Tanzil","TED2020","CCAligned"]
                writer.writerow(firstrow)
                for mod in modifiers:
                    try:
                        with open(f"{path}alldomains_{mod}_{metric}_results.csv", "r") as infile:
                            # print(infile)
                            reader = csv.reader(infile, delimiter="\t")
                            for row in reader:
                                if str(row[0]) == pair:
                                    row.insert(0, mod)
                                    writer.writerow(row)
                                    masterwriter.writerow(row)
                    except FileNotFoundError:
                        row = [mod, pair, "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA"]
                        writer.writerow(row)
                        masterwriter.writerow(row)

