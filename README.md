This Repositary has 4 projects and 1 Final Project.

Text Author identification
•	The goal of this project is to identify the author of a particular piece of text based on the value of perplexity.
•	Uses n-grams along with Laplace smoothing and interpolation for out of vocabulary words
•	Also predicts the likeliest word using a trained model.


Sentiment Analysis
•	Used regular expressions to find the actual words
•	Performed POS tagging using ntlk library
•	Generate features for these words along with the word activeness and pleasantness scores
•	Predict and evaluate the sentiment


POS tagging with HMM
•	Use various types of n-grams to generate feature vector
•	Train the Logistic Regression model over these features
•	Implement Viterbi Algorithm to get the highest probable sequence of tags


Hypernym/Hyponym Relations
•	In this project, we would determine the Hypernym/hyponym relations between two words
•	Developed a simple rule based chunker to identify noun phrases POS tagged sentences
•	Hearst patters are used to achieve the purpose


Final Project:
Machine Translation:
This System translates Hindi sentences to English sentences. We use RNN with LSTM for this purpose. Them we implement Attention at global level and finally implement Byte-Pair Encoding technique to improve the performance in case of unknown/out-of-vocabulary words


