import re
import numpy as np
import math

#output = []


def get_ngrams(n, text):
    output=[]
   # print(type(text))

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

  def update(self, text):

      for i in get_ngrams(self.n, text):

          if i in self.ngram_counts:
              # print("sdgsfg")
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



  def create_ngramlm(self,n, corpus_path):


   cp=self.mask_rare(corpus_path)
   #print(cp)
   listLines = cp.splitlines()
   for i in listLines:
           text = list(i.split())

           get_ngrams(n, text)
           li = self.update(text)

  def word_prob(self, word, context, delta=0):



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
              for j in text:
                  if j in vocab:
                      vocab[j] += 1;
                  else:
                      vocab[j] = 1;

      cp = '<s>'
      with open(corpus_path, 'r') as infile:
          data = infile.readlines()
          text = []
          for i in data:
              text = list(i.split())
              for i in range(0, len(text)):
                  if vocab[text[i]] == 1:
                      text[i] = '<unk>'
              text = ' '.join(text)
              cp = cp + '\n' + text
          # print("cp is ", cp)
          return cp


class NGramInterpolator:
    def __init__(self, n, lambdas):
        self.n = n;
        self.NGramLM_counters=[]
        for i in range(self.n):
            self.NGramLM_counters.append(i)
            self.NGramLM_counters[i] = NGramLM(self.n-i)

        self.lambdas=lambdas;

    def update(self, text):
        for i in range(self.n):
            self.NGramLM_counters[i].update(text)


    #return self.NGramLM_counters[2].count('him')

    def create_ngramlm(self, n, corpus_path):

            #   obj = NGramLM(n)
            cp = self.mask_rare(corpus_path)
            #print(cp)
            #print(self.NGramLM_counters[0])
            listLines = cp.splitlines()
            for i in listLines:
                text = list(i.split())
                # print(text)
                get_ngrams(n, text)
                li = self.update(text)

    def mask_rare(self, corpus_path):
        vocab = {}
        with open(corpus_path, 'r') as infile:
            data = infile.readlines()
            for i in data:
                text = list(i.split())
                for j in text:
                    if j in vocab:
                        vocab[j] += 1;
                    else:
                        vocab[j] = 1;

        cp = '<s>'
        with open(corpus_path, 'r') as infile:
            data = infile.readlines()
            text = []
            for i in data:
                text = list(i.split())
                for i in range(0, len(text)):
                    if vocab[text[i]] == 1:
                        text[i] = '<unk>'
                text = ' '.join(text)
                cp = cp + '\n' + text
            # print("cp is ", cp)
            return cp
        #     t = []
        #     t.append(text[i])
        #     t.append(tuple(text[j:i]))
        #
        #     self.ngram_counts.append(t);
        #
        #     if tuple(text[j:i]) in self.context_counts:
        #         self.context_counts[tuple(text[j:i])] += 1;
        #     else:
        #         self.context_counts[tuple(text[j:i])] = 1;
        #
        #     if text[i] in self.vocabulary:
        #         self.vocabulary[text[i]] += 1;
        #     else:
        #         self.vocabulary[text[i]] = 1;
        #
        # li = [self.ngram_counts, self.context_counts, self.vocabulary]
        # return li;

    def word_prob(self, word, context, delta=0):

        linear_inter_prob=0
        for i in range(self.n):
            linear_inter_prob=linear_inter_prob+ self.NGramLM_counters[i].word_prob(word,context[:self.n-i],delta)*self.lambdas[i]
           # print(word,context,linear_inter_prob)
        return   linear_inter_prob




def text_prob(model, txt,d=0):
    n_grams=get_ngrams(3,txt)
    w_prob=[]
    for i in (n_grams):
      #print(i)
      w_prob.append(model.word_prob(i[0],i[1],d))
    t_prob=1
    for i in range(0,len(w_prob)):
        t_prob=t_prob+math.log(w_prob[i])

    #print(w_prob)
    return t_prob



model = NGramLM(3)

path="/Users/sravyakurra/Desktop/NLP/warpeace.txt"

d=0

txt1="God has given it to me, let him who touches it beware!"
txt2="Where is the prince, my Dauphin?"
txt1=list(txt1.split())
txt2=list(txt2.split())

model.create_ngramlm(3,path)


#print(model.ngram_counts)
print("sentence prob for statement 1:",text_prob(model,txt1,d))

print("sentence prob for statement 2:",text_prob(model,txt2,d))

#
#
# Inter_model = NGramInterpolator(3,[0.33,0.33,0.33])
#
#
#
# #txt1="God has given it to me, let him who touches it beware!"
#
# Inter_model.create_ngramlm(3,path)
# print("Interpolation probability for statement 1",text_prob(Inter_model,txt1,d))
#
# Inter_model.create_ngramlm(3,path)
# print("Interpolation probability for statement 2",text_prob(Inter_model,txt2,d))
#
#




