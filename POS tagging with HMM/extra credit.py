from nltk.corpus import brown
import numpy

from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
import pickle

words = []
rare_words = set()
tag_dict = {}
feature_dict = {}


def main():
    brown_sentences = brown.tagged_sents(tagset='universal')
    # print(brown_sentences)
    train_sentences = []
    train_tags = []
    for i in brown_sentences:
        ts = []
        tt = []
        for j in i:
            ts.append(j[0])
            tt.append(j[1])
        train_sentences.append(ts)
        train_tags.append(tt)
    # print(train_sentences)
    # print(train_tags)

    train_sentences_dictionary = {}
    for i in train_sentences:
        for j in i:
            if train_sentences_dictionary.get(j, 0) == 0:
                train_sentences_dictionary[j] = 1
            else:
                train_sentences_dictionary[j] = train_sentences_dictionary[j] + 1

    for k, v in train_sentences_dictionary.items():
        if v < 5:
            rare_words.add(k)
    # print(len(rare_words))

    training_features = []
    for i in range(len(train_sentences)):
        t = []
        for j in range(len(train_sentences[i])):
            if j == 0:
                prevtag = '<S>'
            else:
                prevtag = train_tags[i][j - 1]
            t.append(get_features(j, train_sentences[i], prevtag, rare_words))
        training_features.append(t)
    print("len of training features", len(training_features))


    remove_result = remove_rare_features(training_features, 5)
    training_features = remove_result[0]

    non_rare_features = remove_result[1]
    counter = 0
    for i in non_rare_features:
        feature_dict[i] = counter
        counter = counter + 1
    count = 0
    for i in range(len(train_tags)):
        for j in range(len(train_tags[i])):
            if tag_dict.get(train_tags[i][j]) is None:
                tag_dict[train_tags[i][j]] = count
                count = count + 1
    print(tag_dict)
    Y_train = build_Y(train_tags)
    # print("xtrain")
    X_train = build_X(training_features)
    # print("is",X_train)
    print(len(Y_train))
    clf_lr = LogisticRegression(class_weight='balanced', solver='saga', multi_class='multinomial')
    clf_lr.fit(X_train, Y_train)
    filename = 'finalized_model.sav'
    pickle.dump(clf_lr, open(filename, 'wb'))
    output = load_test("/Users/sravyakurra/Desktop/NLP/HW3/test.txt")
    tag = []
    for sent in output:
        Y_test = get_predictions([sent],clf_lr)

        tag.append(viterbi(Y_test[1], Y_test[0]))
    print(tag)





def word_ngram_features(i, words):
    if (i) > 0:
        prevbigram = "prevbigram-" + words[i - 1]
    else:
        prevbigram = "prevbigram-<s>"
    if i < len(words) - 1:
        nextbigram = "nextbigram-" + words[i + 1]
    else:
        nextbigram = "nextbigram-</s>"
    if (i - 1) > 0:
        prevskip = "prevskip-" + words[i - 2]
    else:
        prevskip = "prevskip-<s>"
    if i < len(words) - 2:
        nextskip = "nextskip-" + words[i + 2]
    else:
        nextskip = "nextskip-</s>"

    a1 = prevbigram.replace('prevbigram-', '')
    a2 = prevskip.replace('prevskip-', '')
    a3 = nextbigram.replace('nextbigram-', '')
    a4 = nextskip.replace('nextskip-', '')

    prevtrigram = ("prevtrigram-" + a1 + "-" + a2)
    nexttrigram = ("nexttrigram-" + a3 + "-" + a4)
    centertrigram = ("centertrigram-" + a1 + "-" + a3)

    return [prevbigram, nextbigram, prevskip, nextskip, prevtrigram, nexttrigram, centertrigram]


def word_features(word, rare_words):
    output = []
    if word not in rare_words:
        output.append("word-" + word)

    if (any(i.isupper() for i in word)) == True:
        output.append("capital")
    if (any(i.isdigit() for i in word)) == True:
        output.append('number')

    if '-' in word:
        output.append('hyphen')
    for i in range(1, 5):
        if i <= len(word):
            output.append("prefix" + str(i) + "-" + word[:i])
    for j in range(1, 5):
        if j <= len(word):
            output.append("suffix" + str(j) + "-" + word[-j:])
    return output

def remove_rare_features(features, n):
    print("feature len", len(features))
    count_dictionary = {}

    for i in range(len(features)):
        for j in range(len(features[i])):

            for k in features[i][j]:
                if count_dictionary.get(k) is None:
                    count_dictionary[k] = 1
                else:
                    count_dictionary[k] = count_dictionary[k] + 1

    rare_features = set()
    non_rare_features = set()

    for key, v in count_dictionary.items():
        if v < n:
            rare_features.add(key)
        else:
            non_rare_features.add(key)


    for i in range(len(features)):
        for j in range(len(features[i])):

            for k in features[i][j]:
                if k in rare_features:
                    features[i][j].remove(k)

    return [features, non_rare_features]



def get_features(i, words, prevtag, rare_words):

    word_ngram=word_ngram_features(i,words)
    word_feat=word_features(words[i], rare_words)
    tagbigram=["tagbigram-"+prevtag]
    word_prevtag=["word-"+words[i]+"-prevtag-"+prevtag]
    allcaps=[]
    if words[i].isupper():
        allcaps=["allcaps"]
    #wordshape=[]
    digit_flag=0
    word = ''
    for char in words[i]:
        if char.isupper():
            word = word + 'X'
        elif char.islower():
            word = word + 'x'
        elif char.isdigit():
            word = word + 'd'
            digit_flag == 1

    wordshape=["wordshape-" +word]

    k = 0
    str = ""
    word=word+"\n"
    #print(len(word))
    for k in range(len(word)):

        if (k < (len(word) - 1)) and word[k] != word[k + 1]:
            str = str + word[k]

    if len(str) == 0:
        str = word[0]
    short_wordshape=["short-wordshape-"+str]

    a=b=c=0
    if words[i].isupper():
       a=1
    if (any(p.isdigit() for p in words[i])) == True:
        b=1

    if '-' in words[i]:
        c=1
    allcap_dig_hyp=[]
    if a&b&c:
        allcap_dig_hyp=["allcaps-digit-hyphen"]
    res=word_ngram+word_feat+tagbigram
    capital_foll_co=[]
    if words[i][0].isupper():
        for k in range(3):
            if i + k + 1 < len(words) and (words[i + k + 1] == 'Co.' or words[i + k + 1] == 'Inc.'):
             capital_foll_co=['capital-followedby-co']
    res = word_ngram + word_feat + tagbigram + word_prevtag
    if len(allcaps)>0:
        res=res+allcaps
    if  len(allcap_dig_hyp)>0:
        res=res+allcap_dig_hyp
    if len(capital_foll_co)>0:
        res=res+capital_foll_co
    low_ng = [j.lower() for j in res]
   # print("low_ng",type(low_ng))
    low_ng+=wordshape
    low_ng+=short_wordshape
    return low_ng

def build_X(features):
    # new_features = copy.deepcopy(features)
    examples = []
    featuress = []
    values = []
    i = -1
    # print("ffsfsd")
    for feat_sent in features:
        for feat_word in feat_sent:
            i += 1
            for feature in feat_word:
                if feature_dict.get(feature) is not None:
                    # print("csvvfvsf")
                    examples.append(i)
                    temp_ind = feature_dict[feature]
                    featuress.append(temp_ind)
    for ind in range(len(examples)):
        values.append(1)

    examples = numpy.array(examples)
    featuress = numpy.array(featuress)
    values = numpy.array(values)

    matrix = csr_matrix((values, (examples, featuress)), shape=(i + 1, len(feature_dict)))
    return matrix


def build_Y(tags):
    Y = []
    ##print(tag_dict)
    for i in range(len(tags)):
        for j in range(len(tags[i])):
            if tag_dict.get(tags[i][j], 17) != 17:
                Y.append(tag_dict[tags[i][j]])
    return numpy.array(Y)
def load_test(filename):
    li_st = []
    with open(filename, 'r', encoding='latin-1') as infile:
        lines = infile.readlines()
        for i in lines:
            input = list(i.split())
            li_st.append(input)
    return li_st

def get_predictions(test_sentence, model):
    n=len(test_sentence[0])
   # print("n",n)
    T=len(tag_dict)
    Y_pred=numpy.empty([n-1,T,T])
    shape=Y_pred.shape
    print(shape[0])
    #sh=list(shape)

    for i in range(len(test_sentence[0])):
        if i>0:
            #print(test_sentence[0][i])
            for k,v in tag_dict.items():

                feat=get_features(i, test_sentence[0], k, rare_words)

                x_test=build_X([[feat]])

                #print("xtest",x_test.shape)
                Y_pred[i-1][v]=model.predict_log_proba(x_test)

        else:
            feat_start=get_features(i, test_sentence[0], '<S>', rare_words)
            x_test_start = build_X([[feat_start]])
            Y_start=numpy.array(model.predict_log_proba(x_test_start))
    return [Y_pred,Y_start]

def viterbi(Y_start, Y_pred):

    shape=Y_pred.shape
    n=shape[0]+1
    T=len(tag_dict)
    V=numpy.zeros([n,T])
    BP = numpy.zeros([n, T])
    for j in range(T):
        V[0][j]=Y_start[0][j]
        BP[0][j]=-1
    for i in range(n-1):
        for k in range(T):
            V_store=[]
            for j in range(T):

                V_store.append(V[i][j]+Y_pred[i][j][k])

            idx=numpy.argmax(V_store)
           # print("idx",idx)
            V[i+1][k]=V_store[idx]
            BP[i + 1][k]=idx
    backward_indices=[]
    index=numpy.argmax(V[n-1])
    #print("index",index)
    backward_indices.append(index)
    i=n-1
    while i>0:

        index=int(index)
        index=BP[i][index]
        backward_indices.append(index)
        i=i-1
    backward_indices.reverse()
    for idx in range(len(backward_indices)):
        b_idx=backward_indices[idx]
        for key,value in tag_dict.items():
            if value==b_idx:
                backward_indices[idx]=key

    return backward_indices

# def viterbi(Y_start, Y_pred):

if __name__ == '__main__':
    main()