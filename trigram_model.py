import sys
from collections import defaultdict
import math
import random
import os
import os.path
"""
MTH 4335 - Natural Language Processing - Spring 2025
Homework 1 - Trigram Language Models
Howard Yong
Credit - Daniel Bauer, Columbia University
"""

def corpus_reader(corpusfile, lexicon=None): 
    with open(corpusfile,'r') as corpus: 
        for line in corpus: 
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon: 
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else: 
                    yield sequence

def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence: 
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)  



def get_ngrams(sequence, n):
    """
    COMPLETE THIS FUNCTION (PART 1)
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    This should work for arbitrary values of n >= 1 
    """
    result = []

    if (n == 1):
            result.append(tuple(['START']))

    for i in range(len(sequence) + 1):
        n_gram = []
        
        for k in range(n, 0, -1):
            index = i - k + 1
            if index < 0:
                n_gram.append('START')
            elif index == len(sequence):
                n_gram.append('STOP')
            else:
                n_gram.append(sequence[index])

        result.append(tuple(n_gram))

    return result


class TrigramModel(object):
    
    def __init__(self, corpusfile):
    
        # Iterate through the corpus once to build a lexicon 
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")
    
        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)


    def count_ngrams(self, corpus):
        """
        COMPLETE THIS METHOD (PART 2)
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts. 
        """
   
        self.unigramcounts = defaultdict(int)
        self.bigramcounts = defaultdict(int)
        self.trigramcounts = defaultdict(int)
        self.total_num_of_words = 0

        for sequence in corpus:
            n_grams = get_ngrams(sequence, 1) + get_ngrams(sequence, 2) + get_ngrams(sequence, 3)
            self.total_num_of_words += len(sequence)

            for n_gram in n_grams:
                if len(n_gram) == 1:
                    self.unigramcounts[n_gram] += 1
                elif len(n_gram) == 2:
                    self.bigramcounts[n_gram] += 1
                elif len(n_gram) == 3:
                    self.trigramcounts[n_gram] += 1

        return

    def raw_trigram_probability(self,trigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability
        """
        return self.trigramcounts[trigram] / self.bigramcounts[trigram[:2]]

    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        """
        return self.bigramcounts[bigram] / self.unigramcounts[bigram[:1]]
    
    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        """

        #hint: recomputing the denominator every time the method is called
        # can be slow! You might want to compute the total number of words once, 
        # store in the TrigramModel instance, and then re-use it.  
        return self.unigramcounts[unigram] / self.total_num_of_words

    def generate_sentence(self,t=20): 
        """
        COMPLETE THIS METHOD (OPTIONAL)
        Generate a random sentence from the trigram model. t specifies the
        max length, but the sentence may be shorter if STOP is reached.
        """
        return result            

    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation). 
        """
        lambda1 = lambda2 = lambda3 = 1/3.0

        smth_prob = lambda1 * self.raw_trigram_probability(trigram) 
        smth_prob += lambda2 * self.raw_bigram_probability(trigram[:2]) 
        smth_prob += lambda3 * self.raw_unigram_probability(trigram[:1])

        return smth_prob
        
    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """
        logprob = 0

        trigrams = get_ngrams(sentence, 3)

        for trigram in trigrams:
            smth_prob = self.smoothed_trigram_probability(trigram)
            logprob += math.log2(smth_prob)

        return logprob

    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6) 
        Returns the log probability of an entire sequence.
        """
        sum_of_log_probability = 0

        for sentence in corpus:
            sum_of_log_probability += self.sentence_logprob(sentence)

        l = sum_of_log_probability / self.total_num_of_words

        return math.pow(2,-l)


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):

        model1 = TrigramModel(training_file1)
        model2 = TrigramModel(training_file2)

        total = 0
        correct = 0       
 
        for f in os.listdir(testdir1):
            pp = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
            # .. 
    
        for f in os.listdir(testdir2):
            pp = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
            # .. 
        
        return 0.0

if __name__ == "__main__":

    model = TrigramModel(sys.argv[1]) 

    # put test code here...
    # or run the script from the command line with 
    # $ python -i trigram_model.py [corpus_file]
    # >>> 
    #
    # you can then call methods on the model instance in the interactive 
    # Python prompt. 

    
    # Testing perplexity: 
    # dev_corpus = corpus_reader(sys.argv[2], model.lexicon)
    # pp = model.perplexity(dev_corpus)
    # print(pp)


    # Essay scoring experiment: 
    # acc = essay_scoring_experiment('train_high.txt', 'train_low.txt", "test_high", "test_low")
    # print(acc)

