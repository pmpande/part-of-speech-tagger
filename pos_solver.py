###################################
# CS B551 Fall 2015, Assignment #5
#
# Your names and user ids:
# 1. Nishant Shah (nishshah)
# 2. Pranav Pande (pmpande)
#
# (Based on skeleton code by D. Crandall)
#
####--------------------------------------------------------------------------------------------------------------------
# Report:
# 1. Code Description:
# For finding P(S1), P(Si+1|Si) and P(Wi|Si) we have used maximum likelihood estimation.
# P(S1) = count(number of times Si appaered at first position) / count(total number of sentences)
# P(Si+1|Si) = count(Si, Si+1) / count(Si)
# P(Wi|Si) = count(Wi, Si) / count(Si)
# Apart from above we also have dictionary of 1. Words with total count of each word - word[word i]
#                                             2. Part of speech with total count for each POS - part_of_speech[speech i]
#                                             3. Probability of each POS in total POS - prob_speech[speech i]
# For storing probabilities we have used dictionary as in following format
# P(Si+1|Si) - transition[(speech i-1, speech i)]
# P(S1) - initial_probability_distribution[speech i]
# P(Wi|Si) - emission[(word i, speech i)]
#
# Naive:
# Here we are selecting a POS by finding most probable tag for each word.
#
# Sampler:
#
# Max Marginal:
#
# MAP:
# We have first calculated prob. of system in state 0 with each POS. Then finding values for each state by
# multiplying emission value with maximum of product of value of last state and transition prob. from last state to
# current state with each POS. At each state for each POS we are also storing the speech with maximum probability. Thus
# to get find most probable sequence we are selecting values of POS based on POS tagged at current state using its
# stored value.
#
# Best:
# Here we are using Viterbi as our best algorithm. With fined tuned Viterbi(as explained in assumptions section) we are
# getting our best results. For cases where max probability is zero then we infer the word is unknown. Thus assigning
# probability of speech to max probability. This increases percentage by 0.05%(for words) and 0.25%(for sentences)
#
# Comments are provided for each function for better understanding
# ----------------------------------------------------------------------------------------------------------------------
# 2. Results: Scored 2000 sentences with 29442 words.
#                       Words correct:     Sentences correct:
#   0. Ground truth:      100.00%              100.00%
#          1. Naive:       93.92%               47.45%
#        2. Sampler:       18.60%                0.00%
#   3. Max marginal:       18.60%                0.00%
#            4. MAP:       95.69%               57.90%
#           5. Best:       95.74%               58.15%
#
#-----------------------------------------------------------------------------------------------------------------------
#
# 3. Challenges and Assumptions:
#
#   Challenge: Taking decisions for unknown words
#   Assumptions:
#   1. Setting probabilities of unknown words to either zero or a very small value:
#   Here we were facing challenges when any new word comes which is not present in the training corpus. For such words
# we have set their probability to zero(0) or to a very small value(1e-10). In naive while calculating emission prob.
# such words we are assigning its value as zero. Whereas in Viterbi we have assigned emission value and prob. value of
# states as 1e-10 if it is zero.
#   2. Assigning Transition values for unknown transitions to a very small probability:
#   While learning the corpus, if there does not exists any transition from some POS to another then we are assigning
# its value as 1e-10 so that keeping a small probability of its happening.
#
#   Challenge: Fine tuning Viterbi to get best results
#   Assumptions:
#   1. Assigning a very small value when value of the state is zero:
#   If any state is getting its maximum value for the product of last state value and transition value as zero then for
# such states we are assigning its default state value as 1e-10.
#   2. Assigning POS tag of noun when value of state is zero:
#   If any state is getting its maximum value for the product of last state value and transition value as zero then for
# such states we are assigning its default tag as 'noun'.
####--------------------------------------------------------------------------------------------------------------------

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
        sum = 0.0
        multiplcation = 1.0
        for x in range(0, len(sentence)):
            emission_value = self.emission[(sentence[x], label[x])]
            if emission_value == 0:
                emission_value = 1e-10
            sum += emission_value * self.prob_speech[label[x]]
            multiplcation *= emission_value * self.prob_speech[label[x]]
            #hack hack !!! added temporary to stop issue because of no solution for mcmc and marginal.
            #multiplication value becoming 0
            if multiplcation == 0:
                multiplcation = 1e-100
        result = multiplcation/sum

        return math.log(result)

    # Do the training!
    #
    def train(self, data):
        self.transition = {}
        self.emission = {}
        self.word = {}
        self.part_of_speech = {}
        self.transitions = 0
        for row in data:
            #learning from each word of given sentence
            for x in range(0, len(row[0])):
                #Counting frequency of each word
                if row[0][x] not in self.word:
                    self.word[row[0][x]] = 1.0
                else:
                    self.word[row[0][x]] += 1.0
                #Counting frequency of each POS
                if row[1][x] not in self.part_of_speech:
                    self.part_of_speech[row[1][x]] = 1.0
                else:
                    self.part_of_speech[row[1][x]] += 1.0
                #Counting frequency of each POS for appearing at the start of sentence
                if x == 0:
                    if row[1][x] not in self.initial_state_distribution:
                        self.initial_state_distribution[row[1][x]] = 1.0
                    else:
                        self.initial_state_distribution[row[1][x]] += 1.0
                else:
                    inp_string = (row[1][x-1], row[1][x])
                    #Counting frequency of each transition for each POS
                    if inp_string not in self.transition:
                        self.transition[inp_string] = 1.0
                        self.transitions += 1.0
                    else:
                        self.transition[inp_string] += 1.0
                        self.transitions += 1.0
                #Counting frequency of each word with each POS as appearing in the sentences
                if (row[0][x], row[1][x]) not in self.emission:
                    self.emission[(row[0][x], row[1][x])] = 1.0
                else:
                    self.emission[(row[0][x], row[1][x])] += 1.0
        for row in self.word.keys():
            for column in self.part_of_speech.keys():
                #Calculating emission probability
                if (row, column) in self.emission:
                    self.emission[(row, column)] /= self.part_of_speech[column]
        #Calculating initial state distribution
        for x in self.part_of_speech.keys():
            self.initial_state_distribution[x] /= len(data)
        #calculating probability of existence os each POS in the data
        for x in self.part_of_speech.keys():
            self.prob_speech[x] = self.part_of_speech[x] / sum(self.part_of_speech.values())
        #calculating probability for each transition transition(state i-1, state i)
        for x in self.part_of_speech.keys():
            for y in self.part_of_speech.keys():
                if (x, y) not in self.transition:
                    self.transition[(x, y)] = 1e-10
                else:
                    self.transition[(x, y)] /= self.part_of_speech[x]

    # Functions for each algorithm.
    #
    def naive(self, sentence):
        pos = None
        output_list = []
        for word in sentence:
            max_s = 0.0
            for speech in self.part_of_speech.keys():
                if (word, speech) not in self.emission:
                    prob_each = self.emission[(word, speech)] = 0
                else:
                    prob_each = self.emission[(word, speech)] * self.prob_speech[speech]
                #storing max probability and respective POS
                if prob_each > max_s:
                    max_s = prob_each
                    pos = speech
            #if probability is zero then tag 'noun' POS
            if max_s is 0.0:
                pos = 'noun'
            output_list.append(pos)
        return [ [output_list], [] ]

    def mcmc(self, sentence, sample_count):
        return [ [ [ "noun" ] * len(sentence) ] * sample_count, [] ]

    def best(self, sentence):
        output_list = []
        #dictioanary for saving value for states
        v = {}
        #dictionary for saving POS tag while calculating maximum probability for each state
        sequence = {}
        #finding values for state 0 for each POS for first word of the sequence
        for speech in self.part_of_speech.keys():
            if (sentence[0], speech) not in self.emission:
                self.emission[(sentence[0], speech)] = 1e-10
            v[(speech, 0)] = self.initial_state_distribution[speech] * self.emission[(sentence[0], speech)]
        #finding value for each state
        for x in range(1, len(sentence)):
            for speech1 in self.part_of_speech.keys():
                max_prob = 0.0
                max_speech = None
                for speech2 in self.part_of_speech.keys():
                    prob = v[(speech2, x-1)] * self.transition[(speech2, speech1)]
                    if prob > max_prob:
                        max_prob = prob
                        max_speech = speech2
                #if probability is zero then tag 'noun' POS
                if max_prob == 0:
                    max_prob = self.prob_speech[speech1]
                    max_speech = 'noun'
                v[(speech1, x)] = self.emission[(sentence[x], speech1)] * max_prob
                #storing POS with maximum probability for the state
                sequence[(speech1, x)] = max_speech
        tn = len(sentence) - 1
        max = 0.0
        #finding maximum probability for last state
        for speech in self.part_of_speech.keys():
            if v[(speech, tn)] > max:
                max = v[(speech, tn)]
                max_speech = speech
        output_list.append(max_speech)
        tn = len(sentence)
        #tracking values of POS from last state to first by getting saved tag while finding maximum prob. of each state
        # with some POS
        for x in range(0, tn-1):
            speech = sequence[(max_speech, tn - x - 1)]
            max_speech = speech
            output_list.append(speech)

        output_list.reverse()
        return [ [output_list], [] ]



    def max_marginal(self, sentence):
        return [ [ [ "noun" ] * len(sentence)], [[0] * len(sentence),] ]

    def viterbi(self, sentence):
        output_list = []
        #dictioanary for saving value for states
        v = {}
        #dictionary for saving POS tag while calculating maximum probability for each state
        sequence = {}
        #finding values for state 0 for each POS for first word of the sequence
        for speech in self.part_of_speech.keys():
            if (sentence[0], speech) not in self.emission:
                self.emission[(sentence[0], speech)] = 0.001
            prob = self.initial_state_distribution[speech] * self.emission[(sentence[0], speech)]
            v[(speech, 0)] = prob
        #finding value for each state
        for x in range(1, len(sentence)):
            for speech1 in self.part_of_speech.keys():
                max_prob = 0.0
                max_speech = None
                for speech2 in self.part_of_speech.keys():
                    prob = v[(speech2, x-1)] * self.transition[(speech2, speech1)]
                    if prob > max_prob:
                        max_prob = prob
                        max_speech = speech2
                #if probability is zero then tag 'noun' POS
                if max_prob == 0:
                    max_prob = 0.01
                    max_speech = 'noun'
                v[(speech1, x)] = self.emission[(sentence[x], speech1)] * max_prob
                #storing POS with maximum probability for the state
                sequence[(speech1, x)] = max_speech
        tn = len(sentence) - 1
        max = 0.0
        #finding maximum probability for last state
        for speech in self.part_of_speech.keys():
            if v[(speech, tn)] > max:
                max = v[(speech, tn)]
                max_speech = speech
        output_list.append(max_speech)
        tn = len(sentence)
        #tracking values of POS from last state to first by getting saved tag while finding maximum prob. of each state
        # with some POS
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

