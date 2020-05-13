import re
import numpy as np
import math
import random
random.seed(1)

def get_ngrams(n, text):
    output=[]
   # print(type(text))
    if len(text)>0:
        for i in range(n-1):
         text.insert(0, "<s>")
        text.append("</s>")



        for i in range(len(text)):
            j = i - n + 1
            if( tuple((text[i], tuple(text[j:i])))!=("<s>",("<s>","<s>"))):
                 output.append(tuple((text[i], tuple(text[j:i]))));

    return output


class NGramLM:
  def __init__(self, n):
      self.n = n;
      self.ngram_counts = {};
      self.context_counts = {};
      self.vocabulary = {};


  li = []

  def update(self, text):

      for i in get_ngrams(self.n, text):

          if i in self.ngram_counts:

              self.ngram_counts[i] += 1;
          else:
              self.ngram_counts[i] = 1;

          if i[1] in self.context_counts:
              self.context_counts[i[1]] += 1;
          else:
              self.context_counts[i[1]] = 1;

          if i[0] in self.vocabulary:
              self.vocabulary[i[0]] += 1;
          else:
              self.vocabulary[i[0]] = 1;

      li = [self.ngram_counts, self.context_counts, self.vocabulary]
      return li;

  def create_ngramlm(self,n, corpus_path):
   cp=self.mask_rare(corpus_path)
   listLines = cp.splitlines()
   for i in listLines:
       text = list(i.split())
       #get_ngrams(n, text)
       li = self.update(text)

  def word_prob(self, word, context,delta=0):


     if (self.vocabulary.get(word, 0)) == 0:
         word = "<unk>"

     context = list(context)

     for i in range(len(context)):
         if (self.vocabulary.get(context[i], 0)) == 0:
             context[i] = "<unk>"
     context = tuple(context)
     t = []
     # print("out of if",context[0])
     t.append(word)
     t.append(context)

     if self.context_counts.get(tuple(context),0):
        #print("in if")
        w_con=self.context_counts[tuple(context)]

        w_ngram = self.ngram_counts.get(tuple(t), 0)
        prob=(float)(w_ngram+delta)/(w_con+delta*len(self.context_counts.keys()))

        return prob

     else :
        counter = len(self.vocabulary.keys())
        prob=(float)(1/counter)
        #prob=0
        return prob

  def mask_rare(self, corpus_path):
      vocab={}
      with open(corpus_path, 'r') as infile:
          data = infile.readlines()
          for i in data:
              text = list(i.split())
              for j in text:
                  if j in vocab:
                      vocab[j] += 1;
                  else:
                      vocab[j]  = 1;


      cp='<s>'
      with open(corpus_path, 'r') as infile:
          data = infile.readlines()
          text=[]
          for i in data:
                text=list(i.split())
                for i in range(0,len(text)):
                    if vocab[text[i]] ==1:
                        text[i]='<unk>'
                text=' '.join(text)
                cp=cp+'\n'+text
          #print("cp is ", cp)
          return cp

  def random_word (self,context,delta=0 ):
      r=0



      r= (random.random())

      w_p=0
      w_prob_range={}
      tmp=0


      list=sorted(self.vocabulary.keys())


      for i in list:
               tmp =self.word_prob(i, context)
               if(tmp!=0):
                w_prob_range[i]=[w_p,tmp+w_p]
                w_p=tmp+w_p



      res=""
      for k,v in w_prob_range.items():
          if (v[0]<r<v[1])|(v[0]==r):
              res=k
      return  res

  def likeliest_word(self, context, delta=0):
      print(context)
      max = 0
      wd = ""
      list = sorted(self.vocabulary.keys())
      for i in list:
          tmp = self.word_prob(i, context)
         # print(i," ",context," ",tmp)
          if (tmp >= max):
              max = tmp
              wd = i

      return wd



def random_text(model,max_length, delta=0):
 str=" "
 li=[]
 for i in range(model.n -1):
     li.append("<s>")

 for i in range(max_length):

       length = len(li)
       t = tuple(li[length - (model.n) + 1:length])
       #print(t)
       res= model.random_word(t)
       str=str+" "+res
       if(res=="</s>"):
           break
       li.append(res)

 print(str)



def likeliest_text(model, max_length, delta=0):
    str=""
    li=[]
    for i in range(model.n - 1):
        li.append("<s>")
    for i in range(max_length):
            length=len(li)
            t=tuple(li[length-(model.n)+1:length])
            res = model.likeliest_word(t)
            str = str + " " + res
            if (res == "</s>"):
                break
            li.append(res)

    print(str)

def text_prob(model, txt):
    n_grams=get_ngrams(3,txt)
    w_prob=[]
    for i in (n_grams):
     w_prob.append(model.word_prob(i[0],i[1],0.9))
    t_prob=1
    for i in range(0,len(w_prob)):
        t_prob=t_prob+math.log(w_prob[i])
    print("sentence prob",t_prob)



model = NGramLM(5)
path="/Users/sravyakurra/Desktop/NLP/shakespeare_1.txt"

model.create_ngramlm(5,path)

print("Random generation of words")
for i in range(5):
    random_text(model,10)

print("LIKIEST WORD")
likeliest_text(model,10)



