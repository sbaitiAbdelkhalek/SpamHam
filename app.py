"""
@Author: SBAITI Abdelkhalek <abdelkhalek.sbaiti@gmail.com>
"""


import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
import pickle


ps = PorterStemmer()

nltk.download('stopwords')


cv = CountVectorizer()
ps = PorterStemmer()

labelsName = {"ham":0,"spam":1}

model ="null"
filename = './cache/finalized_model.sav'
filenameCV = "./cache/finalize_CountVectorizer.sav"

def preparText(text):
    # Applying Regular Expression

    '''
    Replace email addresses with 'emailaddr'
    Replace URLs with 'httpaddr'
    Replace money symbols with 'moneysymb'
    Replace phone numbers with 'phonenumbr'
    Replace numbers with 'numbr'
    '''
    msg = text
    msg = re.sub('\b[\w\-.]+?@\w+?\.\w{2,4}\b', 'emailaddr', text)
    msg = re.sub('(http[s]?\S+)|(\w+\.[A-Za-z]{2,4}\S*)', 'httpaddr', text)
    msg = re.sub('£|\$', 'moneysymb', text)
    msg = re.sub('\b(\+\d{1,2}\s)?\d?[\-(.]?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b', 'phonenumbr', text)
    msg = re.sub('\d+(\.\d+)?', 'numbr', text)

    ''' Remove all punctuations '''
    msg = re.sub('[^\w\d\s]', ' ', text)

    # Each word to lower case
    msg = msg.lower()

    # Splitting words to Tokenize
    msg = msg.split()

    # Stemming with PorterStemmer handling Stop Words
    msg = [ps.stem(word) for word in msg if not word in set(stopwords.words('english'))]

    # preparing Messages with Remaining Tokens
    msg = ' '.join(msg)
    return msg



def preparingDataForModel(corpus):
    corpusPrepared=[]
    for document in corpus:
        txt = preparText(document)
        corpusPrepared.append(document)
    print("Data was prepared")


    x= cv.fit_transform(corpusPrepared).toarray()
    pickle.dump(cv, open(filenameCV, 'wb'))
    return x



def preparingDataForPreduct(document):
    txt = preparText(document)
    return cv.transform(txt).toarray()



def trainingmodel(numericCorpus,lables,labelsName):

    le = LabelEncoder()
    y = le.fit_transform(lables)

    xtrain, xtest, ytrain, ytest = train_test_split(numericCorpus, y, test_size=0.20, random_state=0)


    print("training is Start")
    model = DecisionTreeClassifier(random_state=50)
    model.fit(xtrain, ytrain)
    print("modele is ready")

    print("test of model")
    # Predicting
    y_pred_dt = model.predict(xtest)

    print("Evaluation of model")
    # Evaluating
    cm = confusion_matrix(ytest, y_pred_dt)

    str ="Accuracy : %0.5f \n\n" % accuracy_score(ytest, model.predict(xtest))+"\n"
    str+=classification_report(ytest, model.predict(xtest))


    pickle.dump(model, open(filename, 'wb'))

    return model,str;



def predict(document):
    vectorizer = pickle.load(open(filenameCV, 'rb'))
    document = preparText(document)
    numericData = vectorizer.transform([document]).toarray()

    #if model == "null":
    model = pickle.load(open(filename, 'rb'))

    result = model.predict(numericData)

    return getLabel(result[0]);





def getLabel(i):
    if i == 0:
        return "ham"
    else:
        return "spam"
