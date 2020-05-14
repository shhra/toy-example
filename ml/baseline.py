#libraries
import pandas as pd 
import csv
import re
import string

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


print("INFO: Loading dataset")
data = pd.read_csv('ISEAR.csv', names=['Number', 'Emotion', 'Sentence'])
data.drop(['Number'], axis=1)

print("INFO: Preprocess start")
#to clean data
def preprocess_data(txt):
    """
    
    Set the sentence in lower
    Remove punctuations from sentence
    
    Parameters
    -------------------
    sentence : str
        all strings of the DataFrame
        
    Returns
    -------------------
    sentence : str
        String without punctuation
            
    """
    txt = re.sub('[^a-zA-Z]', '', txt)
    txt = txt.lower()
    txt = "" . join([w for w in txt if w not in string.punctuation()])
    return txt

print("INFO:Documnet Term matrix")
def doc_term_matrix(df):
    """
    Computes Document Term Matrix
    
    Parameters
    -----------------
    df : pandas.core.frame.DataFrame
        dataset
    
    Returns
    -------------
    dtm : scipy.sparse.csr.csr_matrix
        Sparse document term matrix
        
    """
    data['New_sentence'] = data['Sentence'].apply(lambda x: preprocess_data(x))
    vectorizer = CountVectorizer(stop_words='english', ngram_range=(1,2))
    dtm = vectorizer.fit_transform(data['New_sentence'])
    return dtm


print("INFO: Writing output class file")
def write2file(filename, result_list):
    """
    Write the output class to the text file
    
    Parameters
    -------------
    filename : str
        Output file name
    reurn_list:
        Actual output from classifier
        
    Returns
    -----------
    None
    """
    
    with open(filename, 'w', newline='') as outputfile:
        writer = csv.writer(outputfile)
        for result in result_list:
            writer.writerow([result])


print("INFO: Evaluation start")
def evaluate(y_actual, y_pred):
    """
    
    Evaluates with confusion and accuracy
    
    """
    print(confusion_matrix(y_actual, y_pred))  
    print(accuracy_score(y_actual, y_pred))      

#Splitting the datset
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)   

print("INFO: Model process")
def main():
    
    model = MultinomialNB()
    
    X = doc_term_matrix(data)
    y = data['Emotion']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)   
    model.fit(X_train, y_train)
    
    result_list = model.predict(X_test)
    write2file("bigram.output,txt", result_list)
    
    
if __name__ == " __main__ ":
    main()
    