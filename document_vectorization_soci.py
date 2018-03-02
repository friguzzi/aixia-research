"""
=======================================
Clustering text documents using k-means
=======================================

This is an example showing how the scikit-learn can be used to cluster
documents by topics using a bag-of-words approach. This example uses
a scipy.sparse matrix to store the features instead of standard numpy arrays.

Two feature extraction methods can be used in this example:

  - TfidfVectorizer uses a in-memory vocabulary (a python dict) to map the most
    frequent words to features indices and hence compute a word occurrence
    frequency (sparse) matrix. The word frequencies are then reweighted using
    the Inverse Document Frequency (IDF) vector collected feature-wise over
    the corpus.

  - HashingVectorizer hashes word occurrences to a fixed dimensional space,
    possibly with collisions. The word count vectors are then normalized to
    each have l2-norm equal to one (projected to the euclidean unit-ball) which
    seems to be important for k-means to work in high dimensional space.

    HashingVectorizer does not provide IDF weighting as this is a stateless
    model (the fit method does nothing). When IDF weighting is needed it can
    be added by pipelining its output to a TfidfTransformer instance.

Two algorithms are demoed: ordinary k-means and its more scalable cousin
minibatch k-means.

Additionally, latent semantic analysis can also be used to reduce dimensionality
and discover latent patterns in the data.

It can be noted that k-means (and minibatch k-means) are very sensitive to
feature scaling and that in this case the IDF weighting helps improve the
quality of the clustering by quite a lot as measured against the "ground truth"
provided by the class label assignments of the 20 newsgroups dataset.

This improvement is not visible in the Silhouette Coefficient which is small
for both as this measure seem to suffer from the phenomenon called
"Concentration of Measure" or "Curse of Dimensionality" for high dimensional
datasets such as text data. Other measures such as V-measure and Adjusted Rand
Index are information theoretic based evaluation scores: as they are only based
on cluster assignments rather than distances, hence not affected by the curse
of dimensionality.

Note: as k-means is optimizing a non-convex objective function, it will likely
end up in a local optimum. Several runs with independent random init might be
necessary to get a good convergence.

"""

# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#         Lars Buitinck
# License: BSD 3 clause

from __future__ import print_function
import nltk
import string
import copy
import re
import unicodedata
import csv
from sklearn.externals import joblib
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

from sklearn.cluster import KMeans, MiniBatchKMeans
import glob
import csv
import logging
from optparse import OptionParser
import sys
from time import time

import numpy as np
from os import path
from PIL import Image
import numpy as np

from wordcloud import WordCloud, STOPWORDS

from nltk.corpus import wordnet as wn

d = path.dirname(__file__)

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

# parse commandline arguments
op = OptionParser()
op.add_option("--lsa",
              dest="n_components", type="int",
              help="Preprocess documents with latent semantic analysis.")
op.add_option("--no-minibatch",
              action="store_false", dest="minibatch", default=True,
              help="Use ordinary k-means algorithm (in batch mode).")
op.add_option("--no-idf",
              action="store_false", dest="use_idf", default=False,
              help="Disable Inverse Document Frequency feature weighting.")
op.add_option("--use-hashing",
              action="store_true", default=False,
              help="Use a hashing feature vectorizer")
op.add_option("--n-features", type=int, default=2000,
              help="Maximum number of features (dimensions)"
                   " to extract from text.")
op.add_option("--verbose",
              action="store_true", dest="verbose", default=False,
              help="Print progress reports inside k-means algorithm.")

print(__doc__)
op.print_help()

def removeNonAscii(s): return "".join(i for i in s if ord(i)<128)

def sanitize(w,inputIsUTF8=False,expungeNonAscii=False):

    map =  { 'æ': 'ae',
        'ø': 'o',
        '¨': 'o',
        'ß': 'ss',
        'Ø': 'o',
        '\xef\xac\x80': 'ff',
        '\xef\xac\x81': 'fi',
        '\xef\xac\x82': 'fl',
        u'\u2013': '-',
        'de nition': 'definition'}

  # This replaces funny chars in map
    for char, replace_char in map.items(): 
        w = re.sub(char, replace_char, w)

  #w = unicode(w, encoding='latin-1')
    #if inputIsUTF8: 
        #w = w.encode('ascii', 'ignore')

  # This gets rid of accents
    #w = ''.join((c for c in unicodedata.normalize('NFD', w) if unicodedata.category(c) != 'Mn'))

    #if expungeNonAscii: 
        #w = removeNonAscii(w)

    return w

def gen_cloud(text, cloud_name):

    mask = np.array(Image.open(path.join(d, "logo ai s.ombra 978x532 grey.png")))

    stopwords = set(STOPWORDS)
    stopwords.add("said")
    stopwords.add("model")
    stopwords.add("one")
    stopwords.add("et")
    stopwords.add("al")
    stopwords.add("algorithm")
    stopwords.add("also")
    stopwords.add("given")
    stopwords.add("example")
    stopwords.add("system")
    stopwords.add("user")
    stopwords.add("used")
    stopwords.add("using")
    stopwords.add("use")
    stopwords.add("method")
    stopwords.add("two")
    stopwords.add("node")
    stopwords.add("user")

    wc = WordCloud(background_color=None, mode='RGBA', collocations=False, max_words=2000, mask=mask,
                stopwords=stopwords)
    # generate word cloud
    wc.generate(text)

    # store to file
    wc.to_file(cloud_name)
    return

def gen_cloud_fre(cloud_name,centroid,mask_file):

    mask = np.array(Image.open(path.join(d,mask_file)))


    wc = WordCloud(background_color=None, mode='RGBA', collocations=False,max_words=2000, mask=mask)
    # generate word cloud
    wc.generate_from_frequencies(centroid)

    # store to file
    wc.to_file(cloud_name)
    return


def cluster(opts,true_k,X):

    if opts.minibatch:
        km = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1,
                            init_size=1000, batch_size=1000, verbose=opts.verbose,random_state=None)
    else:
        km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1,
                    verbose=opts.verbose,random_state=None)

    print("Clustering sparse data with %s" % km)
    t0 = time()
    km.fit(X)
    print("done in %0.3fs" % (time() - t0))
    sc=-km.score(X)
    print("score %f" %(sc))
    print()
    if not opts.use_hashing:
        print("Top terms per cluster:")

        order_centroids = km.cluster_centers_.argsort()[:, ::-1]

        terms = vectorizer.get_feature_names()
        
        for i in range(true_k):
            print("Cluster %d:" % i, end='')
            for ind in order_centroids[i, :10]:
                print(' %s' % terms[ind], end='')
            print()
    return (km,sc)

def is_interactive():
    return not hasattr(sys.modules['__main__'], '__file__')

# work-around for Jupyter notebook and IPython console
argv = [] if is_interactive() else sys.argv[1:]
(opts, args) = op.parse_args(argv)
if len(args) > 0:
    op.error("this script takes no arguments.")
    sys.exit(1)

stopWords = set(stopwords.words('english'))
stopWords = stopWords.union(set(stopwords.words('italian')))
stopWords.add("et")
stopWords.add("al")
stopWords.add("wσ")
stopWords.add("riguzzi")
stopWords.add("xijk")
stopWords.add("chk")
stopWords.add("tj")
stopWords.add("symposium")
stopWords.add("esann")
stopWords.add("fig")
stopWords.add("pp")
stopWords.add("let")
stopWords.add("vol")
stopWords.add("figure")
stopWords.add("table")
stopWords.add("proceeding")
stopWords.add("proceedings")
stopWords.add("using")
stopWords.add("use")
stopWords.add("method")
stopWords.add("two")
stopWords.add("node")
stopWords.add("user")
stopWords.add("said")
stopWords.add("model")
stopWords.add("one")
stopWords.add("et")
stopWords.add("al")
stopWords.add("algorithm")
stopWords.add("also")
stopWords.add("given")
stopWords.add("example")
stopWords.add("system")
stopWords.add("user")
stopWords.add("used")
stopWords.add("using")
stopWords.add("use")
stopWords.add("method")
stopWords.add("two")
stopWords.add("node")
stopWords.add("user")
stopWords.add("case")
stopWords.add("time")
stopWords.add("domain")
stopWords.add("state")
stopWords.add("function")
stopWords.add("approach")
stopWords.add("number")
stopWords.add("formula")
stopWords.add("task")
stopWords.add("variable")
stopWords.add("based")
stopWords.add("following")
stopWords.add("obtained")
stopWords.add("problem")
stopWords.add("may")
stopWords.add("section")
stopWords.add("since")
stopWords.add("definition")
stopWords.add("solution")
stopWords.add("point")
stopWords.add("different")
stopWords.add("result")
stopWords.add("show")
stopWords.add("set")
stopWords.add("value")
socio={}
with open('soci.csv', 'rt') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    for row in reader:
        socio[row[2]+' '+row[3]]={'nome':row[2],'cognome':row[3],\
           'filenames':row[2]+' '+row[3],'aff':row[4],'consent':row[5]}
    csvfile.close()

socio.pop("Nome - First Name Cognome - Surname")

socio['Francesca Alessandra Lisi']['filenames']="Francesca A. Lisi"
socio['Fabrizio Riguzzi']['filenames']="Riguzzi Fabrizio"
socio['andrea passerini']['filenames']="Andrea Passerini"
socio['andrea passerini']['nome']="Andrea"
socio['andrea passerini']['cognome']="Passerini"
socio['stefano cagnoni']['filenames']="Stefno Cagnoni"
socio['stefano cagnoni']['nome']="Stefno"
socio['stefano cagnoni']['cognome']="Cagnoni"
socio['Fabrizio Giuseppe Ventola']['filenames']="Fabrizio Ventola"
socio['Daniele Radicioni']['filenames']="daniele radicioni"
socio['cristina cornelio']['filenames']="Cristina Cornelio"
socio['cristina cornelio']['nome']="Cristinao"
socio['cristina cornelio']['cognome']="Cornelio"
socio['Andrea Formisano']['filenames']="formis@dmi.unipg.it"
socio['Salvator Ruggieri']['filenames']="Salvatore Ruggieri"
socio['Salvator Ruggieri']['nome']="Salvatore"
socio['Francesco Ricca']['filenames']="francesco ricca"
socio['Claudio Gallicchio']['filenames']="gallicios"
socio['Cristiano Castelfranchi']['filenames']="cristiano.castelfranchi@istc.cnr.it"
socio['Maria Teresa Pazienza']['filenames']="Teresa pazienza"
socio['AGOSTINO DOVIER']['filenames']="Agostino Dovier"
socio['AGOSTINO DOVIER']['nome']="Agostino"
socio['AGOSTINO DOVIER']['cognome']="Dovier"
socio['Giovanni Maria Farinella']['filenames']="Giovanni Farinella"
socio['Gianluca Torta']['filenames']="gianluca torta"
socio['Daniele Theseider Dupré']['filenames']="Daniele Theseider Dupre'"
socio['stefano cagnoni']['filenames']="Stefano Cagnoni"
socio['stefano cagnoni']['nome']="Stefano"
socio['stefano cagnoni']['cognome']="Cagnoni"
socio['Marco Villani']['filenames']="Marco VILLANI"
socio['Andrea Orlandini']['filenames']="andrea orlandini"
socio['Riccardo Rasconi']['filenames']="riccardo rasconi"
socio['Maria Silvia Pini']['filenames']="Maria Silvia pini"
socio['Gennaro Vessio']['filenames']="Rino Vessio"
socio['PETER SCHUELLER']['filenames']="Peter Schüller"
socio['PETER SCHUELLER']['nome']="Peter"
socio['PETER SCHUELLER']['cognome']="Schüller"
socio['Silvana Badaloni']['filenames']="silvana badaloni"
socio['Stefano Mariani']['filenames']="Stefano MARIANI"
socio['Roberta Calegari']['filenames']="Calegari"
socio['Eva Riccomagno']['filenames']="Luca Banfi"
socio['Marco de Gemmis']['cognome']="De Gemmis"
socio['stefano squartini']['nome']="Stefano"
socio['stefano squartini']['cognome']="Squartini"
socio['Stefano Ferilli']['filenames']="S. Ferilli"
 
for pers in socio:
    socio[pers]['files']=glob.glob("*"+socio[pers]['filenames']+"*")
# for f in socio:
#     if not socio[f]['files']:
#         print(f)

stopWords=list(stopWords)
token_dict = {}
files = glob.glob("*.txt")
files.extend(glob.glob("*.md"))

for file in files:
        shakes = open(file, 'r')
        text = shakes.read()
        shakes.close()
        lowers = text.lower()
        lowers = sanitize(lowers,inputIsUTF8=False,expungeNonAscii=False)
        no_punctuation = lowers.translate(str.maketrans('','',string.punctuation))
#        no_punctuation = no_punctuation.translate(str.maketrans('','', string.printable))
        no_punctuation = ''.join(filter(lambda x: x in string.printable and not x in string.digits, no_punctuation))
#        no_punctuation = no_punctuation.translate({ord(ch): None for ch in string.digits})
        #lowers = sanitize(lowers,inputIsUTF8=True,expungeNonAscii=True)
        # nodigits = lowers.translate({ord(ch): None for ch in string.digits})
        # no_punctuation = nodigits.translate({ord(ch): None for ch in string.punctuation})


        token_dict[file] = no_punctuation


print("%d documents" % len(token_dict.values()))
print()

stemmer = PorterStemmer()
wordnet_lemmatizer = WordNetLemmatizer()
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        if len(item) > 1:
            stemmed.append(wordnet_lemmatizer.lemmatize(item))
    
    return stemmed

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    tokens=list(filter(lambda x: not x in stopWords and len(x)>1 and not wn.synsets(x, pos=wn.ADV),tokens))
    #no_punctuation=''.join(filter(lambda x: wn.synsets(x, pos=wn.ADV), ts))
    stems = stem_tokens(tokens, stemmer)
    # for i in range(len(tokens)-3):
    #     stems.append(tokens[i]+' '+tokens[i+1])
    #     stems.append(tokens[i]+' '+tokens[i+1]+' '+tokens[i+2])
    #     stems.append(tokens[i]+' '+tokens[i+1]+' '+tokens[i+2]+' '+tokens[i+3])
    # i=len(tokens)-3
    # stems.append(tokens[i]+' '+tokens[i+1])
    # stems.append(tokens[i]+' '+tokens[i+1]+' '+tokens[i+2])
    return stems

print("Extracting features from the training dataset using a sparse vectorizer")
t0 = time()
if opts.use_hashing:
    if opts.use_idf:
        # Perform an IDF normalization on the output of HashingVectorizer
        hasher = HashingVectorizer(n_features=opts.n_features,tokenizer=tokenize,
                                   stop_words=stopWords, alternate_sign=False,
                                   ngram_range=(1, 3),
                                   norm=None, binary=False)
        vectorizer = make_pipeline(hasher, TfidfTransformer())
    else:
        vectorizer = HashingVectorizer(n_features=opts.n_features,tokenizer=tokenize,
                                       stop_words=stopWords,ngram_range=(1, 3), 
                                       alternate_sign=False, norm='l2',
                                       binary=False)
else:
    vectorizer = TfidfVectorizer( max_features=opts.n_features,tokenizer=tokenize,
                                max_df=1.0,
                                 min_df=2, stop_words=stopWords,
                                 ngram_range=(1, 3), norm='l1',
                                 use_idf=True)

X = vectorizer.fit_transform(token_dict.values())
joblib.dump(vectorizer, 'vect.pkl') 
joblib.dump(X, 'matrix.pkl') 
