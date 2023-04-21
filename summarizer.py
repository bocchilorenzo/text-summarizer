# coding=utf-8
import numpy as np
from compress_fasttext.models import CompressedFastTextKeyedVectors
from gensim.models import KeyedVectors
from gensim.models.fasttext import load_facebook_model
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine
from nltk.corpus import stopwords
from nltk.tokenize import TreebankWordTokenizer
from nltk.tokenize import sent_tokenize
from re import sub
from os import path
from copy import deepcopy


class LookupTable:
    def __init__(self, model_path, model_type, compressed=True):
        """
        :param model_path: path to the compressed fasttext model
        :param model_type: type of the model to use (fasttext or word2vec)
        :param compressed: True if the model is compressed, False otherwise
        """
        self.model_type = model_type
        if model_type == "word2vec":
            self.model = KeyedVectors.load_word2vec_format(model_path, binary=True, unicode_errors='ignore')
        elif model_type == "fasttext":
            if compressed:
                self.model = CompressedFastTextKeyedVectors.load(path.abspath(model_path))
                self.compressed = True
            else:
                self.model = load_facebook_model(path.abspath(model_path))
                self.compressed = False

    def vec_word(self, word):
        """
        Return the vector of a word if it is in the vocabulary, otherwise return a vector of zeros

        :param word: word to get the vector
        :return: vector of the word
        """
        try:
            if (self.model_type == "fasttext" and self.compressed) or self.model_type == "word2vec":
                return self.model[word]
            elif self.model_type == "fasttext" and not self.compressed:
                return self.model.wv[word]
        except KeyError:
            return np.zeros(1)

    def vec_sentence(self, sentence):
        """
        Return the vector of a sentence if it is in the vocabulary, otherwise return a vector of zeros

        :param sentence: sentence to get the vector
        :return: vector of the sentence
        """
        try:
            return self.model.get_sentence_vector(sentence)
        except KeyError:
            return np.zeros(300)

    def unseen(self, word):
        """
        Check if a word is in the vocabulary

        :param word: word to check
        :return: True if the word is not in the vocabulary, False otherwise
        """
        try:
            if (self.model_type == "fasttext" and self.compressed) or self.model_type == "word2vec":
                self.model[word]
            elif self.model_type == "fasttext" and not self.compressed:
                return not(word in self.model.wv.key_to_index)
            return False
        except KeyError:
            return True


class Summarizer:
    def __init__(
        self,
        model_path=None,
        model_type="fasttext",
        compressed=True,
        tfidf_threshold=0.3,
        redundancy_threshold=0.9,
        language="italian",
        ngram_range=(1, 1),
    ):
        """
        :param model_path: path to the compressed fasttext model
        :param model_type: type of the model to use (fasttext or word2vec)
        :param compressed: True if the model is compressed, False otherwise
        :param tfidf_threshold: threshold to filter relevant terms
        :param redundancy_threshold: threshold to filter redundant sentences
        :param language: language of the text to summarize
        :param ngram_range: range of ngrams to use
        """
        self.lookup_table = LookupTable(model_path, model_type, compressed)
        self.tfidf_threshold = tfidf_threshold
        self.sentence_retriever = []
        self.redundancy_threshold = redundancy_threshold
        self.language = language
        self.ngram_range = ngram_range
        self.model_type = model_type
        self.compressed = compressed

    def _preprocessing(self, text):
        """
        Preprocess the text to summarize

        :param text: text to summarize
        :return: preprocessed text
        """
        # Get splitted sentences
        sentences = self.get_data(text, self.language)

        # Store the sentence before process them. We need them to build final summary
        self.sentence_retriever = deepcopy(sentences)

        # Remove punctuation and stopwords
        sentences = self.remove_punctuation_nltk(sentences)
        sentences = self.remove_stopwords(sentences)

        return sentences

    def _gen_centroid(self, sentences):
        """
        Generate the centroid of the document

        :param sentences: sentences of the document
        :return: centroid of the document
        """
        tf = TfidfVectorizer(ngram_range=self.ngram_range)
        tfidf = tf.fit_transform(sentences).toarray().sum(0)
        tfidf = np.divide(tfidf, tfidf.max())
        words = tf.get_feature_names()

        relevant_terms = [
            words[i]
            for i in range(len(tfidf))
            if tfidf[i] >= self.tfidf_threshold
            and (not self.lookup_table.unseen(words[i]) if self.model_type == "word2vec" else True)
        ]

        res = [self.lookup_table.vec_word(term) for term in relevant_terms]
        return sum(res) / len(res)

    def _sentence_vectorizer(self, sentences):
        """
        Vectorize the sentences of the document

        :param sentences: sentences of the document
        :return: dictionary of sentences vectorized
        """
        dic = {}
        for i in range(len(sentences)):
            sum_vec = np.zeros(self.lookup_table.model.vector_size)
            sentence = [
                word
                for word in sentences[i].split(" ")
                if (not self.lookup_table.unseen(word) if self.model_type == "word2vec" else True)
            ]

            if sentence:
                for word in sentence:
                    word_vec = self.lookup_table.vec_word(word)
                    sum_vec = np.add(sum_vec, word_vec)
                dic[i] = sum_vec / len(sentence)
        return dic

    def _sentence_selection(self, centroid, sentences_dict):
        """
        Select the sentences of the summary
        
        :param centroid: centroid of the document
        :param sentences_dict: dictionary of sentences vectorized
        :return: ids+importance of the sentences of the document, sentences of the document
        """
        record = []
        for sentence_id in sentences_dict:
            vector = sentences_dict[sentence_id]
            similarity = 1 - cosine(centroid, vector)
            record.append((sentence_id, vector, similarity))

        full_ids_importance = [(x[0], x[2]) for x in record]

        full_phrases_list = list(
            map(
                lambda sent_id: self.sentence_retriever[sent_id],
                map(lambda t: t[0], full_ids_importance),
            )
        )
        return full_ids_importance, full_phrases_list

    def summarize(self, text):
        """
        Summarize the text
        
        :param text: text to summarize
        :return: ids+importance of the sentences of the document, sentences of the document
        """
        self._check_params(self.redundancy_threshold, self.tfidf_threshold)

        # Sentences generation (with preprocessing) + centroid generation
        sentences = self._preprocessing(text)

        centroid = self._gen_centroid(sentences)

        # Sentence vectorization + sentence selection
        sentences_dict = self._sentence_vectorizer(sentences)
        ids_importance, phrases = self._sentence_selection(centroid, sentences_dict)

        return ids_importance, phrases

    def remove_punctuation_nltk(self, data):
        """
        Remove punctuation from the sentences
        
        :param data: sentences to process
        :return: sentences without punctuation
        """
        to_return = []
        for sentence in data:
            temp = ""
            tokenized = TreebankWordTokenizer().tokenize(sentence.lower())
            for word in tokenized:
                temp += word
                temp += " "
            to_return.append(temp[:-1])
        return to_return

    def remove_stopwords(self, data):
        """
        Remove stopwords from the sentences
        
        :param data: sentences to process
        :return: sentences without stopwords
        """
        to_return = []
        stop = set(stopwords.words(self.language))
        for sentence in data:
            stopped = ""
            sentence = sentence.lower().split(" ")
            temp = [i for i in sentence if i not in stop]
            for word in temp:
                stopped += word
                stopped += " "
            to_return.append(stopped)
        return to_return

    def get_data(self, text, language):
        """
        Split the text into sentences
        
        :param text: text to split
        :param language: language of the text
        :return: sentences of the text
        """
        sentences = sent_tokenize(text, language)
        fixed_sentences = [self._fix_sentence(s) for s in sentences]
        return fixed_sentences

    @staticmethod
    def _fix_sentence(sentence):
        sentence = sentence.replace("\n", " ")
        sentence = sub(" +", " ", sentence).strip()
        return sentence

    @staticmethod
    def _check_params(redundancy, tfidf):
        try:
            assert 0 <= redundancy <= 1
        except AssertionError:
            raise ("ERROR: the redundancy threshold is not valid")

        try:
            assert 0 <= tfidf <= 1
        except AssertionError:
            raise ("ERROR: the tfidf threshold is not valid")
