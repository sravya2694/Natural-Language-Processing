import re
import numpy as np
import math




def get_ngrams(n, text):
    output=[]

    if len(text) > 0:
        for i in range(n-1):
         text.insert(0, "<s>")
        text.append("</s>")



        for i in range(len(text)):
            j = i - n + 1
            if (tuple((text[i], tuple(text[j:i]))) != ("<s>", ("<s>", "<s>"))):
             output.append(tuple((text[i], tuple(text[j:i]))));

    return output



class NGramLM:
  def __init__(self, n):


    self.n=n;
    self.ngram_counts={};
    self.context_counts={};
    self.vocabulary = {};

  li = []
  def update( self,text):

      for i in get_ngrams(self.n,text):

            if i in self.ngram_counts:

                self.ngram_counts[i] += 1;
            else:
                self.ngram_counts[i] = 1;


            if i[1] in self.context_counts:
                self.context_counts[i[1]]+=1;
            else :
                self.context_counts[i[1]] = 1;

            if i[0] in self.vocabulary:
                self.vocabulary[i[0]]+=1;
            else :
                self.vocabulary[i[0]] = 1;




  def create_ngramlm(self,n, corpus_path):



   with open(corpus_path, 'r') as infile:
       data = infile.readlines()
       for i in data:
           # text = []
           text = list(i.split())
           # print(text)
           get_ngrams(n, text)
           self.update(text)





  def word_prob(self, word, context):
     t = []
     t.append(word)
     t.append(context)


     if self.context_counts.get(tuple(context),0) :

        w_con=self.context_counts[tuple(context)]

        w_ngram=self.ngram_counts.get(tuple(t),0)

        prob=(float)(w_ngram/w_con)

        return prob

     else :

        counter=len(self.vocabulary.keys())

        prob=(float)(1/counter)
        return prob





def text_prob(model, txt):
    n_grams=get_ngrams(3,txt)
    w_prob=[]
    for i in (n_grams):
     # print(i)
      w_prob.append(model.word_prob(i[0],i[1]))
    t_prob=1
    for i in range(0,len(w_prob)):
        t_prob=t_prob+math.log(w_prob[i])

    #print(w_prob)
    return t_prob



n_value=3
model = NGramLM(n_value)



#print(p1.update)
path="/Users/sravyakurra/Desktop/NLP/warpeace.txt"
model.create_ngramlm(n_value,path)

txt1="God has given it to me, let him who touches it beware!"
txt2="Where is the prince, my Dauphin?"
txt1=list(txt1.split())
print("sentence Probability for statement1 is ",text_prob(model,txt1))


txt2=list(txt2.splitlines())
print("sentence Probability for statement2 is ",text_prob(model,txt2))




