import numpy as np
import pandas as pd
import jpype as jp
import time
from collections import Counter
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import f1_score

#Extract the collected dataset and divide into independent and dependent variables
dataset = pd.read_excel('efe_beyazperde_sentiment.xlsx')
X = dataset['text'].str.lower()
slice_X = X.head(3000)
y = dataset['target']
slice_Y = y.head(3000)

# Zemberek Package downloaded and using jpype, JVM has started
# Relative path to Zemberek .jar
ZEMBEREK_PATH = 'C:\\Users\\ehopl\\PycharmProjects\\Efe20199\\VeriUs Staj\\zemberek-nlp\\bin\\zemberek-full.jar'

# Start the JVM
jp.startJVM("C:\\Program Files\\Java\\jdk-12.0.1\\bin\\server\\jvm.dll", "-ea", "-Djava.class.path=%s" % (ZEMBEREK_PATH))

# Import required Java classes
TurkishTokenizer = jp.JClass('zemberek.tokenization.TurkishTokenizer')
TurkishLexer = jp.JClass('zemberek.tokenization.antlr.TurkishLexer')
# Import Morpohology
TurkishMorphology = jp.JClass('zemberek.morphology.TurkishMorphology')
TurkishSpellChecker = jp.JClass('zemberek.normalization.TurkishSpellChecker')

# There are static instances provided for common use:
# DEFAULT tokenizer ignores most white spaces (space, tab, line feed and carriage return).
# ALL tokenizer tokenizes everything.
tokenizer = TurkishTokenizer.DEFAULT

morphology = TurkishMorphology.createWithDefaults()

spell = TurkishSpellChecker(morphology)


#Created a super hero database list with 50+ characters

super_hero_database = ['wolverine', 'hawkeye', 'punisher', 'green goblin', 'storm', 'ghost rider', 'puck', 'gambit', 'war machine', 'dr. strange', 'nightcrawler', 'doc ock', 'spider-man', 'deadpool', 'mystique',
                       'daredevil', 'weapon x', 'cyclops', 'mysterio', 'colosus', 'spider-man (venom kıyafeti)', 'captain america', 'moon knight', 'cable', 'vision', 'bullseye', 'modok', 'black panther', 'sentinel',
                       'plack cat', 'magneto', 'bishop', 'venom', 'beast', 'luke cage', 'archangel', 'scarlet witch', 'ant-man', 'loki', 'thor', 'kraven', 'rogue', 'jubilee', 'red skull', 'iron man', 'red hulk',
                       'iron man (i̇lk tasarım)', 'iron fist', 'hulk', "örümcek adam", "tony stark", "superman", "supergirl", "avengers", "black widow", "kara dul", "wonder woman", "suicide squad", "watchmen"]


# celebrity database!

all_tokens = []
# For every sentence in the reviews, tokenize
for sentence in slice_X:
    characters_found = []
    try:
        for character in super_hero_database:
            if character in sentence:
                sentence = sentence.replace(character, " ")
                characters_found.append(character)
    except:
        pass
    tokens = []
    tokenIterator = tokenizer.getTokenIterator(sentence)

    # Iterating through the tokens using the TokenIterator instance
    while (tokenIterator.hasNext()):
        # Setting the current token
        token = tokenIterator.token
        indexes = [index for (index, letter) in enumerate(str(token)) if letter == "'"]
        str_token = str(token)[indexes[0]+1:indexes[1]]
        if len(str_token)>1 and not(str_token == "...") and not(any(c.isdigit() for c in str_token)):
            if spell.suggestForWord(str_token):
                if not spell.check(str_token):
                    str_token = spell.suggestForWord(str_token)[0]
            try:
                results = morphology.analyze(str_token)
                stem_info = str(results.analysisResults[0])[str(results.analysisResults[0]).index(" ") + 1:]
                stem = stem_info[:stem_info.index(":")]
                tokens.append(stem)
            except:
                pass
            # Printing the token information
            #print('Token = ' + str(token.getText())
            #      + ' | Type (Raw) = ' + str(token.getType())
            #      + ' | Type (Lexer) = ' + TurkishLexer.VOCABULARY.getDisplayName(token.getType())
            #      + ' | Start Index = ' + str(token.getStartIndex())
            #      + ' | Ending Index = ' + str(token.getStopIndex())
            #      )

    tokens.extend(characters_found)
    all_tokens.append(tokens)
dict_array = []
#dataset['tokenized'] = all_tokens
for tok_arr in all_tokens:
    counter = Counter(tok_arr)
    count = {}
    for word in tok_arr:
       if word in count :
          count[word] += 1
       else:
          count[word] = 1
    dict_array.append(count)

vec = DictVectorizer()
X_data = vec.fit_transform(dict_array).toarray()

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_data,slice_Y, test_size=0.2, random_state=0)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=7).fit(X_train, Y_train)

y_pred = knn.predict(X_test)

print(f1_score(Y_test, y_pred, average='macro'))
print(f1_score(Y_test, y_pred, average='micro'))
print(f1_score(Y_test, y_pred, average='weighted'))
print(f1_score(Y_test, y_pred, average=None))

accuracy = knn.score(X_test, Y_test)
print (accuracy)