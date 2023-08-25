import itertools
import bertopic
import numpy as np
import pandas as pd
import advertools as ads
from cuml.manifold import UMAP
from cuml.cluster import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
import langcodes

class TopicModel():
    def __init__(self, **kwargs):

        self.docs = []
        self.src_data = None
        self.ref_data = None
        self.langs = None
        self.stop_langs = None
        self.use_stops = None
        self.stops = None
        self.test_docs = None

        self.n_docs = None
        self.samples = None

        self.embedding_model = None
        self.umap_model = None
        self.hdbscan_model = None
        self.vectorizer_model = None
        self.ctfidf_model = None
        self.topic_model = None

        self.topics, self.probs = None, None
        self.topics_test, self.probs_test = None, None
        self.top_n_idx = None
        self.top_n_probs = None

        self.top_n_topics = None
        self.rep_docs = None

        self.kwargs = kwargs

    def generate_model(self, langs:list, nr_topics:int=100,
                       n_docs:int = 5, samples:int = 1000, src_data=None, ref_data=None,
                       use_stops=False, parallel=False, **kwargs):
        
        # TO DO - combine source and ref sentences
        if parallel:
            self.docs.extend([f"{src} phrase: {src}. {ref} phrase: {ref}" for src, ref in zip(src_data, ref_data)])
        else:
            self.docs.extend(src_data)
            self.docs.extend(ref_data)

        self.langs = langs
        if use_stops:
            self.stop_langs = [langcodes.Language.get(lang).display_name().lower()
                            for lang in self.langs]
            self.stops = self._generate_stops()

        self.n_docs = n_docs
        self.samples = samples

        self.embedding_model = "paraphrase-multilingual-MiniLM-L12-v2"
        self.umap_model = UMAP(n_components=5, n_neighbors=15, min_dist=0.0)
        self.hdbscan_model = HDBSCAN(min_samples=10, gen_min_span_tree=True, prediction_data=True)
        self.vectorizer_model = CountVectorizer(stop_words=self.stops)
        self.ctfidf_model = ClassTfidfTransformer()

        # get language codes - start from codes
        # lgs = [word[0:2] for word in self.langs]

        # for lang in langs:
        #     with open(f"{filepath}{lang}", "r", encoding='utf-8') as file:
        #         fdocs = list(file.readlines())
        #         self.docs.extend(fdocs)

        # init model
        self.topic_model = bertopic.BERTopic(nr_topics=nr_topics,
                        low_memory=True,
                        verbose=False,
                        calculate_probabilities=True,
                        language="multilingual",
                        seed_topic_list=None,
                        top_n_words=10,
                        n_gram_range=(1,3),
                        n_docs = self.n_docs,
                        samples = self.samples,
                        embedding_model=self.embedding_model,
                        umap_model=self.umap_model,
                        hdbscan_model=self.hdbscan_model,
                        vectorizer_model=self.vectorizer_model,
                        ctfidf_model=self.ctfidf_model,
                        **kwargs)

        self.topics, self.probs = self.topic_model.fit_transform(self.docs[0:100])

        return self.topic_model

    def closest_topics(self, sentences:str, top_n:int):

        top_n = -abs(top_n)

        if isinstance(sentences, list):
            pass
        else:
            sentences = [sentences]
        self.test_docs = sentences

        # predict topics for list of docs
        self.topics_test, self.probs_test = self.topic_model.transform(self.test_docs)

        # argsort returns indices to sort array
        self.top_n_idx = [np.flip(np.argsort(doc_probs).flatten()[top_n:]) for doc_probs in self.probs_test]

        self.top_n_probs = [np.flip(np.sort(doc_probs).flatten()[top_n:]).tolist() for doc_probs in self.probs_test]

        topic_dict = self.topic_model.get_topics()
        self.top_n_topics = [{topic_no:topic_dict[topic_no] for topic_no in doc} for doc in self.top_n_idx]

        # redefine top topic (sometimes prediction gives outlier topic -1)
        self.topics_test = [np.flip(np.argsort(doc_probs).flatten()[1:]) for doc_probs in self.probs_test]

        return self.top_n_topics, self.top_n_probs

    def representative_topics(self):

        # must have run closest topics?
        topic_nos = [self.top_n_idx[doc_no][0] for doc_no, doc in enumerate(self.test_docs)]
        self.rep_docs = self.topic_model.get_representative_docs()
        reps = [self.rep_docs[topic_no] for topic_no in topic_nos]

        return reps

    def _generate_stops(self):
        if self.stop_langs:
            stops = [ads.stopwords[lang] for lang in self.stop_langs]
            stops = list(itertools.chain.from_iterable(stops))
        else:
            stops = None
        return stops
    
    def _format_data(self, data_dir):
        for lang in self.langs:
            with open(f"{data_dir}{lang}", "r", encoding='utf-8') as file:
                fdocs = list(file.readlines())
                self.docs.extend(fdocs)


    def _extract_keywords(self):

        pass






