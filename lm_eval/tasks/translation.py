"""
NOTE: This file implements translation tasks using datasets from WMT conferences,
provided by sacrebleu. Traditionally they are evaluated with BLEU scores. TER
and CHRF are other options.

We defer citations and descriptions of the many translations tasks used
here to the SacreBLEU repo from which we've obtained the datasets:
https://github.com/mjpost/sacrebleu/blob/master/sacrebleu/dataset.py

Homepage: https://github.com/mjpost/sacrebleu/blob/master/sacrebleu/dataset.py
"""
import pycountry
from pprint import pprint
from sacrebleu import sacrebleu
from lm_eval import metrics
from lm_eval.base import Task, rf
from typing import List

# EDITS
import os
import itertools
from lm_eval.topicmodel import TopicModel
os.environ["SACREBLEU"] = "/scratch/saycock/data/"

try:
    import nagisa

    HAS_NAGISA = True
except ImportError:
    HAS_NAGISA = False

try:
    import jieba

    HAS_JIEBA = True
except ImportError:
    HAS_JIEBA = False


_CITATION = """
@inproceedings{post-2018-call,
    title = "A Call for Clarity in Reporting {BLEU} Scores",
    author = "Post, Matt",
    booktitle = "Proceedings of the Third Conference on Machine Translation: Research Papers",
    month = oct,
    year = "2018",
    address = "Belgium, Brussels",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/W18-6319",
    pages = "186--191",
}
"""


sacrebleu_datasets = sacrebleu.DATASETS


def create_tasks_from_benchmarks(benchmark_dict):
    """Creates a dictionary of tasks from a dict
    :param benchmark_dict: { dataset: [lang_pair, ...], }
    :return: {task_name: task}
        e.g. {wmt14-fr-en: Task, wmt16-de-en: Task}
    """

    def version_of(dataset, language_pair):
        if language_pair[-2:] in ["zh", "ja"]:
            return 1  # changed to use jieba/nagisa
        return 0

    return {
        f"{dataset}-{language_pair}": create_translation_task(
            dataset, language_pair, version_of(dataset, language_pair)
        )
        for dataset, language_pairs in benchmark_dict.items()
        for language_pair in language_pairs
    }


########################################
# Language Specifics
########################################


def zh_split(zh_text: List[str]) -> List[str]:
    """Chinese splitting"""
    if not HAS_JIEBA:
        raise ImportError(
            "Chinese text splitting requires the `jieba` package. "
            "Please install it with:\npip install jieba"
        )

    return [" ".join(jieba.cut(txt.strip())) for txt in zh_text]


def ja_split(ja_text: List[str]) -> List[str]:
    """Japanese splitting"""
    if not HAS_NAGISA:
        raise ImportError(
            "Japanese text splitting requires the `nagisa` package. "
            "Please install it with:\npip install nagisa"
        )

    return [" ".join(nagisa.tagging(txt.strip()).words) for txt in ja_text]


NO_SPACE_LANG = {"zh": zh_split, "ja": ja_split}

########################################
# Tasks
########################################


def create_translation_task(dataset, language_pair, version=0):
    class TranslationTask(GeneralTranslationTask):
        VERSION = version

        def __init__(self):
            super().__init__(dataset, language_pair)

    return TranslationTask


class GeneralTranslationTask(Task):
    VERSION = 0
    DATA_DIR = "/scratch/saycock/data/"

    # e.g. ("wmt14", "fr-en")
    def __init__(self, dataset, language_pair=None):
        self.sacrebleu_dataset = dataset
        self.data_dir = self.DATA_DIR
        self.sacrebleu_language_pair = language_pair
        self.src_file = self.ref_file = self.src_data = self.ref_data = None
        self.language_codes = self.sacrebleu_language_pair.split("-")

        # EDITS
        self.tm = None

        super().__init__()

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        # This caches in the users home dir automatically
        
        data_dir = self.data_dir

        if "wmt" in self.sacrebleu_dataset:
            self.src_file, self.ref_file = sacrebleu.download_test_set(
                self.sacrebleu_dataset, self.sacrebleu_language_pair
            )
        else:
            self.src_file = data_dir + self.sacrebleu_dataset + "/" + self.sacrebleu_language_pair + "." + self.language_codes[0]
            self.ref_file = data_dir + self.sacrebleu_dataset + "/" + self.sacrebleu_language_pair + "." + self.language_codes[1]

        self.src_data, self.ref_data = [
            [line.rstrip() for line in sacrebleu.smart_open(file)]
            for file in (self.src_file, self.ref_file)
        ]

        # init topic model per dataset
        # TODO: optionally make topic model for all data
        # TODO: add option for topic model to combine source and ref sentences for in-context examples
        self.tm = TopicModel()
        _tm = self.tm.generate_model(langs=self.language_codes,
                                    nr_topics=100,
                                    n_docs=10,
                                    samples=1000,
                                    src_data = self.src_data,
                                    ref_data = self.ref_data,
                                    use_stops=False,
                                    parallel=True)

    def has_training_docs(self):
        """Whether the task has a training set"""
        # TODO In the future we could be more discerning. Some more recent tests have train and dev sets
        return False

    def has_validation_docs(self):
        """Whether the task has a validation set"""
        return False

    def has_test_docs(self):
        """Whether the task has a test set"""
        return True

    def test_docs(self):
        """
        :return: Iterable[obj]
            A iterable of any object, that doc_to_text can handle
        """
        return [
            {"src": src, "ref": ref} for src, ref in zip(self.src_data, self.ref_data)
        ]

    def doc_to_text(self, doc):
        language_codes = self.sacrebleu_language_pair.split("-")
        src_lang = code_to_language(language_codes[0])
        tar_lang = code_to_language(language_codes[1])

        # EDITS
        return f"{src_lang} phrase: " + doc["src"] + f"  {tar_lang} phrase:"

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["src"]

    def doc_to_target(self, doc):
        # This shows a single target, though there may be multiple targets in a lang test
        return " " + doc["ref"] if isinstance(doc["ref"], str) else doc["ref"][0]

    def construct_requests(self, doc, ctx):
        """Uses RequestFactory to construct Requests and returns an iterable of
        Requests which will be sent to the LM.

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param ctx: str
            The context string, generated by fewshot_context. This includes the natural
            language description, as well as the few shot examples, and the question
            part of the document for `doc`.
        """
        return rf.greedy_until(ctx, {"until": ["\n"]})
        # return rf.greedy_until(ctx,  {"max_length": 64})
        # return rf.loglikelihood(ctx)

    def process_results(self, doc, results):
        # Add spaces between words for BLEU score calculation of target languages like Chinese
        tar_lang_code = self.sacrebleu_language_pair.split("-")[-1]
        if tar_lang_code in NO_SPACE_LANG:
            doc["ref"] = NO_SPACE_LANG[tar_lang_code]([doc["ref"]])[0]
            results = NO_SPACE_LANG[tar_lang_code](results)

        # These metrics are corpus-level not sentence level, so we'll hide the
        # results in this dict and compute the corpus score in the aggregate method
        ref_pred = (doc["ref"], results)
        src_ref_pred = (doc["src"], doc["ref"], results)

        return {
            "bleu": ref_pred,
            "chrf": ref_pred,
            # EDITS
            "comet": src_ref_pred
        }

    def aggregation(self):
        """
        :returns: {str: [float] -> float}
            A dictionary where keys are the names of submetrics and values are
            functions that aggregate a list of metrics
        """
        return {
            "bleu": metrics.bleu,
            "chrf": metrics.chrf,
            # EDITS
            "comet": metrics.comet22
        }

    def higher_is_better(self):
        """
        :returns: {str: bool}
            A dictionary where keys are the names of submetrics and values are
            whether a higher value of the submetric is better
        """
        return {
            "bleu": True,
            "chrf": True,
            # EDITS
            "comet": True
        }

    def __str__(self):
        language_codes = self.sacrebleu_language_pair.split("-")
        src_lang = code_to_language(language_codes[0])
        tar_lang = code_to_language(language_codes[1])
        return f"{self.sacrebleu_dataset.upper()} {src_lang} to {tar_lang} Task"

    def fewshot_context(
        self, doc, num_fewshot, provide_description=None, rnd=None, description=None,
        topic_keywords = False, rep_topics = False, no_topics=1
    ):
        """Returns a fewshot context string that is made up of a prepended description
        (if provided), the `num_fewshot` number of examples, and an appended prompt example.

        :param doc: str
            The document as returned from training_docs, validation_docs, or test_docs.
        :param num_fewshot: int
            The number of fewshot examples to provide in the returned context string.
        :param provide_description: bool
            Not implemented, and this option is deprecated and will be removed in a future version in favor of a different description providing method
        :param rnd: random.Random
            The pseudo-random number generator used to randomly sample examples.
            WARNING: This is currently a required arg although it's optionalized with a default `None`.
        :param description: str
            The task's description that will be prepended to the fewshot examples.
        :returns: str
            The fewshot context.
        """
        assert (
            rnd is not None
        ), "A `random.Random` generator argument must be provided to `rnd`"
        assert not provide_description, (
            "The `provide_description` arg will be removed in future versions. To prepend "
            "a custom description to the context, supply the corresponding string via the "
            "`description` arg."
        )
        if provide_description is not None:
            # nudge people to not specify it at all
            print(
                "WARNING: provide_description is deprecated and will be removed in a future version in favor of description_dict"
            )

        description = description + "\n\n" if description else ""
        fewshotex = ""

        if num_fewshot == 0:
            pass
        else:
            # EDITS

            if topic_keywords:
                # best matching topics and probs for given sentence
                n = num_fewshot
                no_topics = num_fewshot
                topics, probs = self.tm.closest_topics(doc["src"], n)
                if no_topics == 1:
                    
                    # TODO: add probability threshold - for sentence's topic probability, not word's prob of being in topic
                    top_topic = self.tm.topics_test[0][0]
                    keyword_text = ("This sentence's predicted topic includes keywords such as: " +
                                ", ".join([keyword_tuple[0] for keyword_tuple in topics[0][top_topic] if keyword_tuple[1] > 0.05])  + ". "
                    )
                else:
                    assert n >= len(topics), "Requested more examples than available topics, please increase no. of closest topics"
                    top_topics = self.tm.topics_test[0][:n].tolist()

                    # get list of all keywords per n topics
                    all_keywords = [topics[0][no] for no in top_topics]
                    all_keywords = list(itertools.chain(*all_keywords))

                    # get list of probabilities per n topics
                    all_probs = [probs[0][c] for c, no in enumerate(top_topics)]

                    # keyword_text = (f"This sentence's best {n} predicted topics include keywords such as:" +
                    #                 ", ".join([keyword_tuple[0] for keyword_tuple, p in zip(all_keywords, all_probs) if p > 0.05])  + ". "
                    #     )
                    
                    keyword_text = (f"This sentence's best {n} predicted topics include keywords such as: " +
                                    ", ".join([keyword for (keyword, p) in all_keywords if p > 0.05])  + ". "
                        )

                fewshotex += keyword_text
            
            if rep_topics:
                n = num_fewshot
                no_topics = num_fewshot
                topics, probs = self.tm.closest_topics(doc["src"], n)
                # list of sentences that represent topic
                rep = self.tm.representative_topics()[0]

                rep_text = ("Representative sentences in the closest topic include: " +
                            "\n".join([r for r in rep])
                )

                fewshotex += rep_text

            # # for sets with no training docs, draw from other set *but ensure no overlap with current doc*
            # if self.has_training_docs():
            #     fewshotex = self.fewshot_examples(k=num_fewshot, rnd=rnd)
            # else:
            #     if self._fewshot_docs is None:
            #         self._fewshot_docs = list(
            #             self.validation_docs()
            #             if self.has_validation_docs()
            #             else self.test_docs()
            #         )

            #     fewshotex = rnd.sample(self._fewshot_docs, num_fewshot + 1)

            #     # get rid of the doc that's the one we're evaluating, if it's in the fewshot
            #     fewshotex = [x for x in fewshotex if x != doc][:num_fewshot]

        # labeled_examples = (
        #     "\n\n".join(
        #         [
        #             self.doc_to_text(doc) + self.doc_to_target(doc)
        #             for doc in fewshotex
        #         ]
        #     )
        #     + "\n\n"
        # )

        labeled_examples = fewshotex

        example = self.doc_to_text(doc)

        return labeled_examples + example


########################################
# Util
########################################


def code_to_language(code):
    # key is alpha_2 or alpha_3 depending on the code length
    language_tuple = pycountry.languages.get(**{f"alpha_{len(code)}": code})
    return language_tuple.name
