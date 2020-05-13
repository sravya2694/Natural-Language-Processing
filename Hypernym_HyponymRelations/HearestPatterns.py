import nltk
import re
import sys


# Fill in the pattern (see Part 2 instructions)

NP_grammar = 'NP: {<DT>?<JJ.?>*<NN.?>+}'


# Fill in the other 4 rules (see Part 3 instructions)
hearst_patterns = [
    ('(NP_\w+ (, NP_\w+)* (, )?(and|or) other NP_\w+ )', 'after'),
    ('(NP_\w+ (, )?such as (NP_\w+ ? (, )?(and |or )?)+)', 'before'),
    ('(such NP_\w+ as (NP_\w+ ? (, )?(and |or )?)+)', 'before'),
    ('(NP_\w+ (, )?including (NP_\w+ ? (, )?(and |or )?)+)', 'before'),
    ('(NP_\w+ (, )?especially (NP_\w+ ? (, )?(and |or )?)+)', 'before')
 ]


# Fill in the function (see Part 1 instructions)
# Argument type: path - string
# Return type: list of tuples
def load_corpus(path):
    result = []
    with open(path, 'r', encoding='latin-1') as infile:
        lines = infile.readlines()
        for i in lines:
            input = list(i.split("\t"))
            sent = list(input[0].split())
            lem_sent = list(input[1].split())
            result.append((sent, lem_sent))
    return result


# Fill in the function (see Part 1 instructions)
# Argument type: path - string
# Return type: tuple of sets
def load_test(path):
    true = set()
    false = set()
    with open(path, 'r', encoding='latin-1') as infile:
        lines = infile.read().splitlines()

        for i in lines:
            input = list(i.split("\t"))

            if (input[2].strip() == "True"):
                true.add((input[0].strip(), input[1].strip()))
                j = 1
            elif (input[2].strip() == "False"):
                false.add((input[0].strip(), input[1].strip()))

    return (true, false)


# Fill in the function (see Part 2 instructions)
# Argument type: sentence, lemmatized - list of strings; parser - nltk.RegexpParser
# Return type: string
def chunk_lemmatized_sentence(sentence, lemmatized, parser):
    res = nltk.pos_tag(sentence)
   # print(res)
    lem_res = []
    for i in range(len(sentence)):
        lem_res.append((lemmatized[i], res[i][1]))


    result = parser.parse(lem_res)
   # print(result)

    chunks = tree_to_chunks(result)
    string = merge_chunks(chunks)
    string=string+" "
    #print(string)
    return string


# Fill in the function (see Part 2 instructions)
# Helper function for chunk_lemmatized_sentence()
# Argument type: tree - nltk.Tree
# Return type: list of strings
def tree_to_chunks(tree):
    chunks = []
    for child in tree:

        if isinstance(child, nltk.Tree):
           # print(child.leaves())
            tk = []
            for i in child.leaves():
                tk.append(i[0])
            np = '_'.join(tk)
            np = 'NP_' + np
            chunks.append(np)
        else:
            chunks.append(child[0])
   # print(chunks)
    return chunks


# Fill in the function (see Part 2 instructions)
# Helper function for chunk_lemmatized_sentence()
# Argument type: chunks - list of strings
# Return type: string
def merge_chunks(chunks):
    buffer = []
    for chunk in chunks:
        if len(buffer) == 0:
            buffer.append(chunk)
        else:
            if ("NP_" in buffer[-1]) & ("NP_" in chunk):
                buffer[-1] = buffer[-1] + "_" + chunk.replace("NP_", "")
            else:
                buffer.append(chunk)
    buffer = " ".join(buffer)
    return buffer


# Fill in the function (see Part 4 instructions)
# Argument type: chunked_sentence - string
# Yield type: tuple
def extract_relations(chunked_sentence):
   # chunked_sentence = 'NP_European_Countries , especially NP_France_England_Spain . '
      #what if a sentance matches more than 1 pattern
   # print(chunked_sentence)
    result=[]
    for pattern in hearst_patterns:
        matched_list = []
        #print(pattern)
        m=re.search(pattern[0],chunked_sentence)
        #print(m)
        if m is not None:
            matched=m.group(0)
            #print("matched is",matched)
            matched_list=matched.split()
        #print("matched list is",matched_list)
        new = []
        if len(matched_list)>0:

            for i in range(len(matched_list)):
            # print(list[i])
                if "NP_" in matched_list[i]:
                    new.append(matched_list[i])

           # print(new)
            list=postprocess_NPs(new)
            #print(list)
            if pattern[1]=="before":

                for i in range(1,len(list)):
                    result.append((list[i],list[0]))

                    #print((list[i],list[0]))
            elif pattern[1]=="after":
                for i in range(len(list)-1):
                    result.append((list[i],list[-1]))

   # print("result is",result)
    return result


# Fill in the function (see Part 4 instructions)
# Helper function for extract_relations()
# Argument type: list of strings
# Return type: list of strings
def postprocess_NPs(NPs):
    new=[]
    for i in range(len(NPs)):
           str= NPs[i].replace("NP_","")
           new.append( str.replace("_"," "))
    #print(new)
    return new


# Fill in the function (see Part 5 instructions)
# Argument type: extractions, gold_true, gold_false - set of tuples
# Return type: tuple
def evaluate_extractions(extractions, gold_true, gold_false):
    # print(gold_true)
    #print(gold_true)
    #print(extractions)

    tp = 0
    fp = 0
    fn = 0
    a = set()

    for tup in extractions:
        if tup in gold_true:
            print("true positive")
            tp = tp + 1
            a.add(tup)


        elif tup in gold_false:
            fp = fp + 1
            print("False positive")

    fn = len(gold_true) - len(a)
    print(tp)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f_measure = 2 * precision * recall / (precision + recall)
    return precision, recall, f_measure


def main(args):
    corpus_path = args[0]
    test_path = args[1]


    #parser = nltk.RegexpParser(grammar)
    #load_corpus("/Users/sravyakurra/Desktop/NLP/HW4/wikipedia_sentences.txt")
    #load_test("/Users/sravyakurra/Desktop/NLP/HW4/test.tsv")


    wikipedia_corpus = load_corpus(corpus_path)
    test_true, test_false = load_test(test_path)

    NP_chunker = nltk.RegexpParser(NP_grammar)

    list=[]
    for sent in wikipedia_corpus:
        list.append(chunk_lemmatized_sentence(sent[0],sent[1],NP_chunker))
    # Complete the line (see Part 2 instructions)
    wikipedia_corpus = list
    #print("wiki corpus",wikipedia_corpus)

    extracted_pairs = set()
    for chunked_sentence in wikipedia_corpus:
        for pair in extract_relations(chunked_sentence):
            extracted_pairs.add(pair)
   # evaluate_extractions(extracted_pairs, test_true, test_false)
    #
    print('Precision: %f\nRecall:%f\nF-measure: %f' % evaluate_extractions(extracted_pairs, test_true, test_false))
    #evaluate_extractions("()", test_true, test_false)

if __name__ == '__main__':
    #print(sys.argv[1:])
    sys.exit(main(sys.argv[1:]))
