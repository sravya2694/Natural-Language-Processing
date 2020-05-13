import operator
import re;
import nltk;
import numpy as np;
import sys
#np.set_printoptions(threshold=sys.maxsize)
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression


def regexpr(text):
    regexp=re.findall(r"\bnever\b|\bnot\b|\bno\b|\bcannot\b|\b[a-zA-Z]+n't\b",text)

    return regexp

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

    snippet=re.sub(r"(^|\s)(')([^']+?)(')($|\s)", r"\1\2 \3 \4 \5", snippet)
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
        if regexpr(List_wordpos[i][0])!=[]:


                b=True
                if i+1<len(List_wordpos) :
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



def get_features(preprocessed_snippet):
    v3=score_snippet(preprocessed_snippet,dal)
    feature_vector=np.zeros(len(feature_dict)+3)
    #print(feature_vector)
    for i in preprocessed_snippet:
        if "EDIT_" not in i[0]:
            if i[0]  in feature_dict:

                num=feature_dict.get(i[0])
                feature_vector[num]=feature_vector[num]+1
    length=len(feature_vector)


    feature_vector[-3] = v3[0]
    feature_vector[-2] = v3[1]
    feature_vector[-1] = v3[2]

    return feature_vector

def normalize(X):
   shp=X.shape
   print(shp)
   #return X
   for i in range(shp[1]):

        max=X[np.argmax(X[:,i])][i]
        min=X[np.argmin(X[:,i ])][i]
       # print(max," ",min)
        #print(X[min][i])
        for j in range(shp[0]):
            f=X[j][i]
            if max==min:
                X[j][i]=f-min
            else:

             X[j][i]=(f-min)/(max-min)



   return X


def top_features(clf1,k):
    # a = []
    # c = []
    # print(clf1.coef_)
    List_wt = (clf1.coef_).tolist()[0]

    List_idx = []

    for i in range(len(List_wt)):
        List_idx.append(tuple((i,List_wt[i])))



    #print(List_idx)
    tup_wt_desc=sorted(List_idx, key=lambda x: abs(x[1]),reverse=True)

   # print(tup_wt_desc)
    #List_wt_desc=list(tup_wt_desc)
    List_wt_desc=[]

    for i in range(len(tup_wt_desc)):
        Li=[]
        for key,v in feature_dict.items():
            if v==tup_wt_desc[i][0]:
                Li.append(key)
        if len(Li)==0:
            if tup_wt_desc[i][0]==len(feature_dict):
                Li.append("activeness_dal")
            if tup_wt_desc[i][0] == len(feature_dict) + 1:
                Li.append("evaluation_dal")
            if tup_wt_desc[i][0] == len(feature_dict) + 2:
                Li.append("imager_dal")
        Li.append(tup_wt_desc[i][1])
        List_wt_desc.append(tuple(Li))
    #print(k," ",type(k))
    r_list=[]
    for i in range(k):
        print(List_wt_desc[i])
        r_list.append(List_wt_desc[i])
    return r_list




def load_dal(dal_path):
    dal_dict={}
    with open(dal_path, 'r', encoding='latin-1') as infile:
        next(infile)

        data = infile.readlines()
        for i in data:
            text = list(i.split("\t"))

            dal_dict[text[0]]=tuple((float(text[1]),float(text[2]),(float)(text[3])))

    return dal_dict


def score_snippet(preprocessed_snippet, dal):
    #print(preprocessed_snippet)
    li_act=[]
    li_pl=[]
    li_img=[]
    for i in preprocessed_snippet:
        if "EDIT_" not in i[0]:
            if "NOT_" in i[0]:
                key = i[0].replace("NOT_", "")
                if key in dal:
                    val=dal[key]
                    li_act.append(-1*val[0])
                    li_pl.append(-1*val[1])
                    li_img.append(-1*val[2])
                else:
                    li_act.append(0)
                    li_pl.append(0)
                    li_img.append(0)

            else :
                key = i[0]
                if i[0] in dal:
                    val=dal[key]
                    li_act.append(val[0])
                    li_pl.append(val[1])
                    li_img.append(val[2])
                else:

                    li_act.append(0)
                    li_pl.append(0)
                    li_img.append(0)



    # print(li_act)
    # print(li_pl)
    # print(li_img)
    return (sum(li_act)/len(li_act),sum(li_pl)/len(li_pl),sum(li_img)/len(li_img))






def evaluate_predictions(Y_pred, Y_true):
    tp=fp=fn=0
    for i in range(len(Y_true)):
        if ((Y_pred[i]) == 1) & ((Y_true[i] )== 1):
               tp=tp+1
        if ((Y_true[i])==0 )& ((Y_pred[i])==1):
                 fp=fp+1
        if ((Y_true[i])==1)  & ((Y_pred[i])==0):
              fn=fn+1
    print(tp," ",fp, " ",fn)
    precision=(float)(tp/(tp+fp))
    recall = (float)(tp / (tp + fn))
    f_measure = (float)(2*precision*recall/(precision+recall))
    print(precision, " ", recall, " ", f_measure)
    return (precision,recall,f_measure)


dal=load_dal("/Users/sravyakurra/Desktop/NLP/HW@2/dict_of_affect.txt")
Li=load_corpus("/Users/sravyakurra/Desktop/NLP/HW@2/train.txt")
#print(len(Li))
for i in Li:
    #print(i)

    List=tokenize(i[0])


    index=len(feature_dict)
    for i in List:
            if "EDIT_" not in i[0]:
                if i[0] not in feature_dict:
                    feature_dict[i[0]] = index
                    index = index + 1
    #length=len(feature_dict)

X_train=np.empty(shape=(len(Li),len(feature_dict)+3))
Y_train=np.empty(shape=len(Li))

for i in range(len(Li)):
    #print(Li[i])
    List = tokenize(Li[i][0])
#    score_snippet(List,dal)
    X_train[i]=get_features(List)
    Y_train[i]=Li[i][1]
# for i in range(len(Li)):
#     #print(Li[i])
#     print(X_train[i],":",Y_train[i])

normalized_X=(normalize(X_train))
clf = GaussianNB()
clf.fit(normalized_X, Y_train)
clf_lr=LogisticRegression()
clf_lr.fit(normalized_X, Y_train)


Li_test=load_corpus("/Users/sravyakurra/Desktop/NLP/HW@2/test.txt")
X_test=np.empty(shape=(len(Li_test),len(feature_dict)+3))
Y_test=np.empty(shape=len(Li_test))


for i in range(len(Li_test)):
    #print(Li[i])
    List = tokenize(Li_test[i][0])

    X_test[i]=get_features(List)
    Y_test[i] = Li_test[i][1]
#print(X_test)
#print("hiii")

normalized_Xtest = (normalize(X_test))

#Y_pred=clf.predict(normalized_Xtest)
Y_pred_lr=clf_lr.predict(normalized_Xtest)
#print(Y_pred)
#print(Y_test)


evaluate_predictions(Y_pred_lr,Y_test)

print("top featurtes:",top_features(clf_lr,10))