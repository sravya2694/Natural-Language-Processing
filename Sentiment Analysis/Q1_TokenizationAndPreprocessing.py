import operator
import re;
import nltk;
import numpy as np;
import sys
#np.set_printoptions(threshold=sys.maxsize)
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

regexp=re.compile(r"(^|\s)(')([^']+?)(')($|\s)")
neg_regexp=re.compile(r"\bnever\b|\bnot\b|\bno\b|\bcannot\b|\b[a-zA-Z]+n't\b")




feature_dict={}


def load_corpus(corpus_path):

    li=[]
    with open(corpus_path, 'r',encoding='latin-1') as infile:
        data = infile.readlines()
        for i in data:

            text = list(i.split("\t"))
            snippet=text[0]
            label=(int)(text[1])
            li.append((snippet,label))

    return li
List=[]
vocab=set()
def tokenize(snippet):

    snippet=regexp.sub( r"\1\2 \3 \4 \5", snippet)
    snippet=snippet.split()
    tagged=tag_edits(snippet)
    List=(tag_negation(tagged))
    return List

def tag_edits(tokenized_snippet):
    start=end=0
    for i in range(len(tokenized_snippet)):
        if '[' in tokenized_snippet[i]:
           start=i
        if ']' in tokenized_snippet[i]:
            end=i
            break
    if start!=end:
        for i in range(start,end+1):
            if i==start:
                tokenized_snippet[i]=tokenized_snippet[i].replace("[","")
            elif i==end:
                tokenized_snippet[i] = tokenized_snippet[i].replace("]", "")

            if tokenized_snippet[i]=="":
                tokenized_snippet.remove("")
            tokenized_snippet[i]='EDIT_'+tokenized_snippet[i]
    return (tokenized_snippet)

    #         while(1):
    #             if ']' in tokenized_snippet[i]:
    #                 break;
    #             tokenized_snippet[i]='EDIT_'+tokenized_snippet[i];
    # print(tokenized_snippet)

stop_tag_words={"but","however","nevertheless",".","?","!"}

def tag_negation(tokenized_snippet):
    copy=tokenized_snippet[:]
    #print(copy)
    #print(tokenized_snippet)
    for i in range(len(tokenized_snippet)):
        if "EDIT_" in tokenized_snippet[i]:
            tokenized_snippet[i] = tokenized_snippet[i].replace("EDIT_", "")

    wordpos=nltk.pos_tag(tokenized_snippet)


    List_wordpos=list(wordpos)


    for i in range(len(copy)):
        if "EDIT_" in copy[i]:
            li=list(List_wordpos[i])
            li[0]="EDIT_"+li[0]
            List_wordpos[i] = tuple(li)


    #print(List_wordpos)
    b=False

    for i in range(len(List_wordpos)):
      #  print(i)
        if neg_regexp.findall(List_wordpos[i][0])!=[]:


                b=True
                if  List_wordpos[i][0]=="not":
                    if i+1<len(List_wordpos):
                        if List_wordpos[i+1][0]=="only":
                         b=False
                continue
        if (List_wordpos[i][0] in stop_tag_words)|(List_wordpos[i][1] in['JJR','RBR']):
           # print()
            b=False
        if b==True:
            li = list(List_wordpos[i])
            li[0] = "NOT_" + li[0]
            List_wordpos[i] = tuple(li)


    return (List_wordpos)

    #nltk.pos_tag()





#dal=load_dal("/Users/sravyakurra/Desktop/NLP/HW@2/dict_of_affect.txt")
Li=load_corpus("/Users/sravyakurra/Desktop/NLP/HW@2/train.txt")
#print(len(Li))
for i in Li:
    #print(i)

    List=tokenize(i[0])
    print(List)


