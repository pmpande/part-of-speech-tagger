# part-of-speech-tagger
Implemented a part of speech tagger, using Bayesian networks, in which the goal was to mark every word in a sentence with its correct part of speech. Provided with a large dataset based on Brown corpus for training and testing purpose. Developed using techniques – Naïve inference, Gibbs Sampling and Viterbi algorithm.

This project is based on skeleton code by PROF. DAVID CRANDALL.

The files contained in this project specifies information of authors for each method or function written in that file.

Contributers to this project:
1. PRANAV PANDE 
2. NISHANT SHAH 

# Code Description

For finding P(S1), P(Si+1|Si) and P(Wi|Si) we have used maximum likelihood
P(S1) = count(number of times Si appaered at first position) / count(total number of sentences), 

P(Si+1|Si) = count(Si, Si+1) / count(Si),

P(Wi|Si) = count(Wi, Si) / count(Si). 

Apart from above we also have dictionary of 1. Words with total count of each word - word[word i], 
                                            2. Part of speech with total count for each POS - part_of_speech[speech i], 
                                            3. Probability of each POS in total POS - prob_speech[speech i]. 
                                            
 For storing probabilities we have used dictionary as in following format
 
 P(Si+1|Si) - transition[(speech i-1, speech i)], 
 
 P(S1) - initial_probability_distribution[speech i], 
 
 P(Wi|Si) - emission[(word i, speech i)]. 

 Naive:
 Here we are selecting a POS by finding most probable tag for each word.

 Sampler:
 We have used Gibbs sampling of MCMC method as asked. We are starting with initial sample having all POS tag being
 'noun' as 'noun' has highest probability among other POS. We are sampling 20 more samples than whichever asked as we
 will discard first 5.

 Max Marginal:
 It's almost same as MCMC except we are taking more samples and taking max marginal probability for POS tag instead
 randomly generating.

 MAP:
 We have first calculated prob. of system in state 0 with each POS. Then finding values for each state by
 multiplying emission value with maximum of product of value of last state and transition prob. from last state to
 current state with each POS. At each state for each POS we are also storing the speech with maximum probability. Thus
 to get find most probable sequence we are selecting values of POS based on POS tagged at current state using its
 stored value.

 Best:
 Here we are using Viterbi as our best algorithm. With fined tuned Viterbi(as explained in assumptions section) we are
 getting our best results. For cases where max probability is zero then we infer the word is unknown. Thus assigning
 probability of speech to max probability.

 Comments are provided for each function for better understanding
 
# Results

  Results: Scored 2000 sentences with 29442 words.

                        Words correct:     Sentences correct:
    1. Ground truth:      100.00%              100.00%
    2. Naive:             92.89%               43.40%
    3. Sampler:           93.65%               46.30%
    4. Max marginal:      94.74%               53.80%
    5. MAP:               95.09%               54.50%
    6. Best:              95.09%               54.50%


# Challenges and Assumptions:

  Challenge: Taking decisions for unknown words
   
  Assumptions:
  1. Setting probabilities of unknown words to a very small value:
  Here we were facing challenges when any new word comes which is not present in the training corpus. For such words
  we have set their probability to a very small value(1e-10). In naive approach while calculating emission prob. of
  such words we are assigning its value as zero. In Viterbi we have assigned emission value and prob. value of
  states as 1e-10 if it is zero. But we haven't missed to do renormalization.
   
  2. Assigning Transition values for unknown transitions to a very small probability:
  While learning the corpus, if there does not exists any transition from some POS to another then we are assigning
  its value as 1e-10 so that keeping a small probability of its happening.

  Challenge: Fine tuning Viterbi to get best results
  
  Assumptions:
  1. Assigning a very small value when value of the state is zero:
  If any state is getting its maximum value for the product of last state value and transition value as zero then for
  such states we are assigning its default state value as 1e-10.
   
  2. Assigning POS tag of noun when value of state is zero:
  If any state is getting its maximum value for the product of last state value and transition value as zero then for
  such states we are assigning its default tag as 'noun'.

