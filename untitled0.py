

from sklearn.neural_network import MLPClassifier
from bs4 import BeautifulSoup as bs
import codecs
from nltk import word_tokenize
import os


##############################vvvvvvvvvvvvvvvvvv Modify for all files in folder
dr = 'E:\Wikipedia\Test Set'

filelist = os.listdir(dr)
finalstring = ''

for data in filelist:
    data = codecs.open(dr + "\\" + data, encoding='utf-8')
    soup = bs(data, 'html.parser')
    
    soup = soup.find_all('p')#find data we're looking for in file, returns array

    print("extracting ", data)
    for item in soup:
        finalstring = finalstring + str(item) + ' '#combine all array items into 1 string
#######################^^^^^^^^^^^^^^^^^^^^

print("parsing")
soup = bs(finalstring, 'html.parser')#parse string into normal english
del finalstring #lots of data, gott keep memory clean

print("tokenizing")
tokens = word_tokenize(soup.get_text())#break string into invidual words
tokens = [w.lower() for w in tokens]#make all words lowercase


print("removing worthless words")
i = len(tokens) - 1
for item in reversed(tokens): #get rid of worthless 'words'
    if (item == '(' or item == ')' or item == '.' or item == '“' or item == '”' \
        or item == ','or item=='}' or item=='--'):  
        del tokens[i]
    i = i - 1
    
    
print("sorting words")
tokensSorted = sorted(list(tokens))#sort words
del item
bag = []#bag of words, list of words and their commonality


print("counting word usage")
j = -1#keeping track of our bag of words
for word in tokensSorted:
    if i != -1:
        if tokensSorted[i] != word:
            bag.append([word, 1])
            j = j + 1 
        else:
            bag[j][1] = bag[j][1] + 1
    else:
        bag.append([word, 1])
        j = j + 1
    i = i + 1
del j, i, word, tokensSorted

print("sorting words")
bag.sort(key = lambda x: x[1], reverse = True)
commonlist = []#vocabulary and Dictionary

print("simplifying dictionary")
for item in bag:
    commonlist.append(item[0])
    
del bag, item# we dont need the number of uses anymore, so we can get rid of it

dataset = []
targets = []
print("generating data set")
for i in range(len(tokens)-5):#creating our dataset
    temp = []
    for j in range(6):
        for k in range(len(commonlist)):
            if commonlist[k] == tokens[i +(j)]:
                if j != 5:
                    temp.append(k)
                else:
                    targets.append(k)
    dataset.append(temp)

    
    ######################## DATA FINALLY PREPARED WOOO!!######################

print("preparing neural network")
mlp = MLPClassifier(max_iter = 15000000)
mlp.fit(dataset, targets)


def predict():
    startString = [[1,2,3,4,5]]
    print("Enter your string now:")
    predictString = input()
    predictString = word_tokenize(predictString)
    
    print("generating\n\n")
    for j in range(len(predictString)):
        for i in range(len(commonlist)):
            if commonlist[i] == predictString[j]:
                del startString[0][0]
                startString[0].append(i)
                
    
    prediction = []
    
    for i in range(50):
        prediction.append(mlp.predict(startString))
        del startString[0][0]
        startString[0].append(prediction[i][0])
    
    for item in predictString:
        print(item + ' ', end='')
        
    for item in prediction:
        print(commonlist[item[0]] + ' ', end='')
        
