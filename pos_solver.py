###################################
# CS B551 Fall 2015, Assignment #5
#
# Your names and user ids:
#
# (Based on skeleton code by D. Crandall)
#
#
####
# Put your report here!!
####

import random
import math

# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#
class Solver:

    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling
    transition = {}
    emission = {}
    initial_state_distribution = {}
    prob_speech = {}
    part_of_speech = {}
    transitions = 0
    word = {}

    def posterior(self, sentence, label):
        return 0

    # Do the training!
    #
    def train(self, data):
        self.transition = {}
        self.emission = {}
        self.word = {}
        self.part_of_speech = {}
        self.transitions = 0
        for row in data:
            for x in range(0, len(row[0])):

                if row[0][x] not in self.word:
                    self.word[row[0][x]] = 1.0
                else:
                    self.word[row[0][x]] += 1.0

                if row[1][x] not in self.part_of_speech:
                    self.part_of_speech[row[1][x]] = 1.0
                else:
                    self.part_of_speech[row[1][x]] += 1.0

                if x == 0:
                    if row[1][x] not in self.initial_state_distribution:
                        self.initial_state_distribution[row[1][x]] = 1.0
                    else:
                        self.initial_state_distribution[row[1][x]] += 1.0
                else:
                    inp_string = (row[1][x-1], row[1][x])

                    if inp_string not in self.transition:
                        self.transition[inp_string] = 1.0
                        self.transitions += 1.0
                    else:
                        self.transition[inp_string] += 1.0
                        self.transitions += 1.0

                if (row[0][x], row[1][x]) not in self.emission:
                    self.emission[(row[0][x], row[1][x])] = 1.0
                else:
                    self.emission[(row[0][x], row[1][x])] += 1.0
        for row in self.word.keys():
            for column in self.part_of_speech.keys():
                if (row, column) in self.emission:
                    self.emission[(row, column)] /= self.part_of_speech[column]
        for x in self.part_of_speech.keys():
            self.initial_state_distribution[x] /= len(data)
        for x in self.part_of_speech.keys():
            self.prob_speech[x] = self.part_of_speech[x] / sum(self.part_of_speech.values())
        for x in self.part_of_speech.keys():
            for y in self.part_of_speech.keys():
                if (x, y) not in self.transition:
                    self.transition[(x, y)] = 1e-10
                else:
                    self.transition[(x, y)] /= self.part_of_speech[x]
        print "Training Complete"
    # Functions for each algorithm.
    #
    def naive(self, sentence):
        max_s = None
        pos = None
        output_list = []
        for word in sentence:
            max_s = 0.0
            for speech in self.part_of_speech.keys():
                if (word, speech) not in self.emission:
                    prob_each = self.emission[(word, speech)] = 0
                else:
                    prob_each = self.emission[(word, speech)] * self.prob_speech[speech]
                if prob_each > max_s:
                    max_s = prob_each
                    pos = speech
            if max_s is 0.0:
                pos = 'noun'
            output_list.append(pos)
        return [ [output_list], [] ]

    def mcmc(self, sentence, sample_count):
        return [ [ [ "noun" ] * len(sentence) ] * sample_count, [] ]

    def best(self, sentence):
        return [ [ [ "noun" ] * len(sentence)], [] ]

    def max_marginal(self, sentence):
        return [ [ [ "noun" ] * len(sentence)], [[0] * len(sentence),] ]

    def viterbi(self, sentence):
        output_list = []
        v = {}
        sequence = {}
        for speech in self.part_of_speech.keys():
            if (sentence[0], speech) not in self.emission:
                self.emission[(sentence[0], speech)] = 1e-10
            prob = self.initial_state_distribution[speech] * self.emission[(sentence[0], speech)]
            v[(speech, 0)] = prob
        for x in range(1, len(sentence)):
            for speech1 in self.part_of_speech.keys():
                max_prob = 0.0
                max_speech = None
                for speech2 in self.part_of_speech.keys():
                    prob = v[(speech2, x-1)] * self.transition[(speech2, speech1)]
                    if prob > max_prob:
                        max_prob = prob
                        max_speech = speech2
                if max_prob == 0:
                    max_prob = 1e-10
                    max_speech = 'noun'
                v[(speech1, x)] = self.emission[(sentence[x], speech1)] * max_prob
                sequence[(speech1, x)] = max_speech
        tn = len(sentence) - 1
        max = 0.0
        for speech in self.part_of_speech.keys():
            if v[(speech, tn)] > max:
                max = v[(speech, tn)]
                max_speech = speech
        output_list.append(max_speech)
        tn = len(sentence)
        for x in range(0, tn-1):
            speech = sequence[(max_speech, tn - x - 1)]
            max_speech = speech
            output_list.append(speech)

        output_list.reverse()
        return [ [output_list], [] ]


    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself. 
    # It's supposed to return a list with two elements:
    #
    #  - The first element is a list of part-of-speech labelings of the sentence.
    #    Each of these is a list, one part of speech per word of the sentence.
    #    Most algorithms only return a single labeling per sentence, except for the
    #    mcmc sampler which is supposed to return 5.
    #
    #  - The second element is a list of probabilities, one per word. This is
    #    only needed for max_marginal() and is the marginal probabilities for each word.
    #
    def solve(self, algo, sentence):
        if algo == "Naive":
            return self.naive(sentence)
        elif algo == "Sampler":
            return self.mcmc(sentence, 5)
        elif algo == "Max marginal":
            return self.max_marginal(sentence)
        elif algo == "MAP":
            return self.viterbi(sentence)
        elif algo == "Best":
            return self.best(sentence)
        else:
            print "Unknown algo!"

