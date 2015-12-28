###################################
# CS B551 Fall 2015, Assignment #5
#
# Your names and user ids:
# 1. Nishant Shah (nishshah)
# 2. Pranav Pande (pmpande)
#
# (Based on skeleton code by Prof. David Crandall)

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
    #S is list of length qual to highest numbers of words is any sentence in training data.
    #Each element is a dictionary having every POS as its key and values as a tuple containing total number of occurence
    #of key in training data and it's probability for that index of list.
    S = []

    #Returns a list having dictionary elements. Member dictionary elements have have values as tuple containing total
    # number of occurences of key in training data and it's probability.

    # Author of the following method: Nishant Shah
    def prob(self,prior_data):
        for i in range(0,len(prior_data)):
            total = sum(prior_data[i].values())
            for key in prior_data[i].keys():
                prior_data[i][key] = (prior_data[i][key],prior_data[i][key] / total)
        return prior_data

    # Author: Pranav Pande (Based on skeleton code by Prof. David Crandall)
    def posterior(self, sentence, label):
        sum = 0.0
        multiplcation = math.log(1)
        for x in range(0, len(sentence)):
            emission_value = self.emission[(sentence[x], label[x])]
            if emission_value == 0:
                emission_value = 1e-10
            sum += emission_value * self.prob_speech[label[x]]
            multiplcation += math.log(emission_value * self.prob_speech[label[x]])
            result = multiplcation - math.log(sum)
        return result


    # Do the training!
    # Author of the following method: Pranav Pande (Based on skeleton code by Prof. David Crandall)
    def train(self, data):
        self.transition = {}
        self.emission = {}
        self.word = {}
        self.part_of_speech = {}
        self.transitions = 0

        for row in data:
            #learning from each word of given sentence
            for x in range(0, len(row[0])):
                #S is list of length qual to highest numbers of words is any sentence in training data.
                try:
                    self.S[x][row[1][x]] = self.S[x].get(row[1][x],0) + 1.0
                except IndexError:
                    self.S += [{row[1][x]:1.0}]
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
        #Calculating emission probability
        for row in self.word.keys():
            for column in self.part_of_speech.keys():
                if (row, column) in self.emission:
                    self.emission[(row, column)] /= self.part_of_speech[column]
        #Calculating initial state distribution
        for x in self.part_of_speech.keys():
            self.initial_state_distribution[x] /= len(data)
        #calculating probability of existence of each POS in the data
        for x in self.part_of_speech.keys():
            self.prob_speech[x] = self.part_of_speech[x] / sum(self.part_of_speech.values())
        #calculating probability for each transition transition(state i-1, state i)
        for x in self.part_of_speech.keys():
            for y in self.part_of_speech.keys():
                if (x, y) not in self.transition:
                    self.transition[(x, y)] = 1e-10
                else:
                    self.transition[(x, y)] /= self.part_of_speech[x]
        #If any word for a given position doesn't have some POS tagged in whole training data, we assume it to have
        #occured 0.00001 time. Normalization is done later in 'prob' function
        for index in range(0,len(self.S)):
            if len(self.part_of_speech.keys()) != len(self.S[index].keys()):
                for pos in set(self.part_of_speech.keys()) - set(self.S[index].keys()):
                    self.S[index][pos] = 0.00001
        self.S = self.prob(self.S)

    # Functions for each algorithm.
    # Author of the following method: Pranav Pande (Based on skeleton code by Prof. David Crandall)
    def naive(self, sentence):
        pos = None
        output_list = []
        for word in sentence:
            max_s = 0.0
            for speech in self.part_of_speech.keys():
                if (word, speech) not in self.emission:
                    prob_each = self.emission[(word, speech)] = 1e-10
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

    # Author of the following method: Nishant Shah (Based on skeleton code by Prof. David Crandall)
    def mcmc(self, sentence, sample_count):
        #S is a list of length equal to total words in given sentence and it stores POS for each word in sentence.
        S = ['noun']*len(sentence)
        #samples stores each sample generated.
        samples = [S]
        #To discard initial 20 samples we are sampling total sample_count+20 samples
        for n in range(0,sample_count + 20):
            for i in range(0,len(S)):
                #Stored the probablity distribution for each for each POS
                S1_prob_dist = []
                S1_pos_dist = []
                #for 1st word
                if i==0:
                    #Sampling over all possible POS for given position and taking probability of each
                    for pos in self.prob_speech.keys():
                        #If any of the probability can not be derived training dataset, we take it as 0.001 and we do
                        #not nomalize it here as we do not store them in database. And it's also not needed as we are
                        #doing normalization later
                        try:
                            #S1: P(s_1)*P(s_2 | S_1)*P(w_1 | S_1).
                            #self.S = [{pos: (pos_count,pos_prob}]
                            S1_prob_dist += [self.S[0][pos][1]*self.transition.get((pos,S[1]),0.001)*self.emission.get((sentence[0],pos),0.001)]
                            S1_pos_dist += [pos]
                        except IndexError:
                            #Exception for a sentence with single word
                            S1_prob_dist += [self.S[0][pos][1]*self.emission.get((sentence[0],pos),0.001)]
                            S1_pos_dist += [pos]
                #For last word
                elif i== len(S)-1:
                    for pos in self.prob_speech.keys():
                        S1_prob_dist += [self.transition.get((S[i-1],pos),0.001)*self.emission.get((sentence[-1],pos),0.001)]
                        S1_pos_dist += [pos]
                else:
                    for pos in self.prob_speech.keys():
                     #P(s_i | s_{i-1}), P(s_{i+1} | s_i), and P(w_i | s_i).
                        S1_prob_dist += [self.transition.get((S[i-1],pos),0.001)*self.transition.get((pos,S[i+1]),0.001)*self.emission.get((sentence[i],pos),0.001)]
                        S1_pos_dist += [pos]
                temp = 0.0
                #doing normalization and also randomaly generating probability and assigning appropriate POS from CPT
                #generated from samples
                rand_temp = random.random()
                for j in range(0,len(S1_prob_dist)):
                    temp += S1_prob_dist[j]/sum(S1_prob_dist)
                    if rand_temp <= temp:
                        S[i] = S1_pos_dist[j]
                        break
            samples += [S]
        temp = []
        #Returning samples from reverse which is given in parameter of function call
        for i in samples[-sample_count:]:
            temp.append(i)
        return [ temp, [] ]

    # Author of the following method: Pranav Pande (Based on skeleton code by Prof. David Crandall)
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

    #Max_marginal is same as MCMC except we are taking 1000 samples. Also we tag POS to a word having maximum_marginal
    #value instea radomly generating probability.
    # Author of the following method: Nishant Shah (Based on skeleton code by Prof. David Crandall)
    def max_marginal(self, sentence):
        S = ['noun']*len(sentence)
        #Stored marginal probability of each POS for a word
        max_marg = [0.0]*len(sentence)
        samples = [[S,max_marg]]
        #To discard initial 20 samples we are sampling total sample_count+20 samples
        for n in range(0,1000):
            for i in range(0,len(S)):
                S1_prob_dist = []
                S1_pos_dist = []
                if i==0:
                    #S1: P(s_1)*P(s_2 | S_1)*P(w_1 | S_1).
                    #self.S = [{pos: (pos_count,pos_prob}]
                    #Sampling over all possible POS for given position and taking probability of each
                    for pos in self.prob_speech.keys():
                        try:
                            S1_prob_dist += [self.S[0][pos][1]*self.transition.get((pos,S[1]),0.001)*self.emission.get((sentence[0],pos),0.001)]
                            S1_pos_dist += [pos]
                        except IndexError:
                            S1_prob_dist += [self.S[0][pos][1]*self.emission.get((sentence[0],pos),0.001)]
                            S1_pos_dist += [pos]
                elif i== len(S)-1:
                    for pos in self.prob_speech.keys():
                        S1_prob_dist += [self.transition.get((S[i-1],pos),0.001)*self.emission.get((sentence[-1],pos),0.001)]
                        S1_pos_dist += [pos]
                else:
                    for pos in self.prob_speech.keys():
                     #P(s_i | s_{i-1}), P(s_{i+1} | s_i), and P(w_i | s_i).
                        S1_prob_dist += [self.transition.get((S[i-1],pos),0.001)*self.transition.get((pos,S[i+1]),0.001)*self.emission.get((sentence[i],pos),0.001)]
                        S1_pos_dist += [pos]
                temp = 0.0
                #Max_marg differs from MCMC from here
                #Taking max conditional probablity from derived samples for each word
                j = max(S1_prob_dist)
                #Taking POS having max conditional probability
                S[i] = S1_pos_dist[S1_prob_dist.index(j)]
                #Normalizing
                max_marg[i] = round(j/sum(S1_prob_dist),2)
            #Just saving last sample generated unlike MCMC
            samples = [[S,max_marg]]
        return [ [samples[-1][0]], [samples[-1][1]] ]

    # Author of the following method: Pranav Pande (Based on skeleton code by Prof. David Crandall)
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
    # Author of the following method: Prof. David Crandall
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

