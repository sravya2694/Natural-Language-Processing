import re
import numpy as np
import math

#output = []


def get_ngrams(n, text):
    output=[]
   # print(type(text))
    if len(text) > 0:
        for i in range(n):
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
                #print("sdgsfg")
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

      li=[self.ngram_counts,self.context_counts,self.vocabulary]
      return li;


  def create_ngramlm(self,n, corpus_path):


   cp=self.mask_rare(corpus_path)

   listLines = cp.splitlines()
   for i in listLines:

           # text = []
           text = list(i.split())
           # print(text)
           get_ngrams(n, text)
           li = self.update(text)
   print("vocab count after masking:",len(self.vocabulary.keys()))



  def word_prob(self, word, context, delta=0):


      if (self.vocabulary.get(word, 0)) == 0:
          word = "<unk>"

      context = list(context)

      for i in range(len(context)):
          if (self.vocabulary.get(context[i], 0)) == 0:
              context[i] = "<unk>"

      # if(self.context_counts.get(tuple(context),0))==0:
      #     context= "<unk>"
      context = tuple(context)
      t = []
      # print("out of if",context[0])
      t.append(word)
      t.append(context)

      if self.context_counts.get(tuple(context), 0):
          # print("in if")
          w_con = self.context_counts[tuple(context)]

          w_ngram = self.ngram_counts.get(tuple(t), 0)
          prob = (float)(w_ngram + delta) / (w_con + delta * len(self.context_counts.keys()))

          return prob

      else:
          counter = len(self.vocabulary.keys())
          prob = (float)(1 / counter)
          # prob=0
          return prob

  def mask_rare(self, corpus_path):
      vocab = {}
      with open(corpus_path, 'r') as infile:
          data = infile.readlines()
          for i in data:

                  text = list(i.split())
                  if text!=" ":
                      for j in text:
                          if j in vocab:
                              vocab[j] += 1;
                          else:
                              vocab[j] = 1;

      cp = '<s>'
      print("vocab count before mask",len(vocab.keys()))
      with open(corpus_path, 'r') as infile:
          data = infile.readlines()
          text = []
          for i in data:
              text = list(i.split())

              for i in range(len(text)):
                  if vocab[text[i]] == 1:
                      text[i] = '<unk>'
              text = ' '.join(text)
              cp = cp +'\n'+ text
          # print("cp is ", cp)
          return cp




def text_prob(model, txt,delta=0):
    n_grams=get_ngrams(3,txt)
    w_prob=[]
    for i in (n_grams):
     # print(i)
      w_prob.append(model.word_prob(i[0],i[1],delta))
    t_prob=0
    for i in range(0,len(w_prob)):
        t_prob=t_prob+math.log(w_prob[i])

    # print(w_prob)
    # print("senstance prob",t_prob)
    return t_prob

def perplexity(model, corpus_path,delta=0):
    total_text_prob=0
    counter=0
    with open(corpus_path, 'r') as infile:
        data = infile.readlines()
        for i in data:
            # text = []
            if i!=" ":
                text = list(i.split())
                counter=counter+len(text)
                sen_prob=text_prob(model,text,delta)
                total_text_prob=total_text_prob+sen_prob
    total_text_prob=total_text_prob/counter
    print("total_text_prob ",total_text_prob)
    pp=math.exp(-total_text_prob)
    print("PP is",pp)


  #
model = NGramLM(3)

delta=0.5

#print(p1.update)
path="/Users/sravyakurra/Desktop/NLP/warpeace.txt"

model.create_ngramlm(3,path)
corpus_path="/Users/sravyakurra/Desktop/NLP/sonnets.txt"
perplexity(model, corpus_path,delta)



