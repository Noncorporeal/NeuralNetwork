

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from bs4 import BeautifulSoup as bs
import codecs
from nltk import word_tokenize
##############################vvvvvvvvvvvvvvvvvv Modify for all files in folder

data = codecs.open('E:\Wikipedia\ZZZ data set\Poland.html', encoding='utf-8')

soup = bs(data, 'html.parser')

soup = soup.find_all('p')#find data we're looking for in file, returns array
finalstring = ''

for item in soup:
    finalstring = finalstring + str(item) + ' '#combine all array items into 1 string
#######################^^^^^^^^^^^^^^^^^^^^
    

soup = bs(finalstring, 'html.parser')#parse string into normal english
del finalstring #lots of data, gott keep memory clean

tokens = word_tokenize(soup.get_text())#break string into invidual words
tokens = [w.lower() for w in tokens]#make all words lowercase



i = len(tokens) - 1
for item in reversed(tokens): #get rid of worthless 'words'
    if (item == '(' or item == ')' or item == '.' or item == '“' or item == '”' or item == ','):  
        del tokens[i]
    i = i - 1
    
    
    
tokensSorted = sorted(list(tokens))#sort words
del item
bag = []#bag of words, list of words and their commonality



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


bag.sort(key = lambda x: x[1], reverse = True)
commonlist = []

for item in bag:# we dont need the number of uses anymore, so we can get rid of it
    commonlist.append(item[0])
    
del bag, item

dataset = []
targets = []

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

train_data, test_data, train_target, test_target = train_test_split(dataset, targets, train_size=0.8, test_size=0.2, random_state=321)

mlp = MLPClassifier()
mlp.fit(train_data, train_target)

prediction = mlp.predict(test_data)



i = 0
for item in prediction:
    outputs = ''
    for thing in test_data[i]:
        outputs = outputs + commonlist[thing] + ' '
    outputs = outputs + commonlist[item]
    print(outputs)
    i += 1



