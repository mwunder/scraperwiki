import numpy as np
from numpy import *
import sqlite3
import csv
import operator
import math
import random
import time
import re
import heapq
import nltk
import itertools
import pickle
from nltk.corpus import stopwords
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
from nltk.tokenize import word_tokenize, wordpunct_tokenize, sent_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.corpus import wordnet as wn

class rv:
    def __init__(self, idIn, prodIn, catIn, ratingIn, reviewIn):
        self.id = idIn
        self.product = prodIn
        self.category = catIn
        self.rating = ratingIn
        self.review = reviewIn
        self.taggedPOS = []

categoryName = 'tvs'
inputFilename = 'reviews_'+categoryName+'.csv'

# Find the stem of a word using the Lancaster stemmer algorithm.
# When two words have the same stem, ie 'taste' and 'tasting' they can be considered the same.
st = LancasterStemmer()  # SnowballStemmer('english') #

# Efficient part-of-speech tagger. 
tagger = nltk.data.load("taggers/treebank_brill_aubt.pickle")

# Operations with wordnet are expensive, so we should keep a common
# list of valid sets of synonyms to speed up operations instead of reconstructing it.
try:
    f = open('synonyms.pickle')
    synonyms_dict = pickle.load(f)
    f.close()
    if len(synonyms_dict)==0:
        synonyms_dict = {}
except:
    synonyms_dict = {}
changed_synonyms = set()

# Some of the synonyms actually mean the opposite of what we want.
# For example, 'great' can mean very good, but also big. And for some
# reason 'bad' is classified as a synonym of big, so let's remove these cases. 
nonsynonyms ={'big':['bad'],'bad':['big','great'],'great':['bad']}

# Some words are generic and uninformative, and we don't want them showing up as features.
nonfeatures = ['product', 'like', 'look','make', 'use', 'work', 'get', 'way','recommend']

# Some commonly used words, plus some conjunctions.  These usually don't signify qualities so let's ignore them. 
stopwords = nltk.corpus.stopwords.words('english')
stopwords = stopwords + ["n't",'ive','youre','didnt','wouldnt','wasnt','isnt','theyre','im','id',\
                         'hadnt','would','could','should','cant','wont','dont','havent','thats','youll',\
                         'isnt',"'m","'re","'s","'t","'ve","also"]

# Signifiers for negations.
notwords = ["not","never","no","n't"]

# Given a csv file where columns are separated by ',\N,', extract data into a
# list of reviews.
def extract_reviews(filename,limit):
    rvs = []
    f = open(filename)
    for line in f:
        if limit<=0:
            break
        else:
            if len(rvs)>0 and (len(rvs[-1].review)==0 or rvs[-1].review.count(',')>len(rvs[-1].review)/2):
                del rvs[-1]
            else:
                limit-=1
        data = line.split(',\N,')
        try:
            if len(data[4])>0 and data[4].find('|')<0:
                category_this = data[0].split('/')
                rvs.append(rv(0,data[1],category_this[-1].lower(),int(data[3]),data[4].lower()))
        except:
            if data[0]!='name' and data[0].find('|')<0:
                rvs[-1].review += data[0].lower()
    f.close()
    return rvs

def extract_from_db(conn,limit):
    rvs = []
    return rvs

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

# Return a mapping from stem to the most common full word from that stem.
# This is how we decide which one to use.
def common_full_words(review_text):
    stem_count = {}
    stem_map = {}
    for review in review_text:
        for w in [word for word in review if word not in stopwords and len(word)>2 and not is_number(word)]:
            try:
                wstem = st.stem(w)
            except:
                wstem = st.stem(unicode(w,errors='ignore'))
            try:
                stem_count[wstem][w] += 1
            except:
                try: 
                    stem_count[wstem][w] = 1
                except: 
                    stem_count[wstem] = {}
                    stem_count[wstem][w] = 1
    for (wstem,wstem_count) in stem_count.items():
        if len(wstem_count)>1:
            max_count = 0
            max_word = ''
            for w in wstem_count.keys():
                if wstem_count[w]>max_count:
                    max_count = wstem_count[w]
                    max_word = w
            stem_map[wstem] = max_word
        else:
            stem_map[wstem] = wstem_count.keys()[0]
    return stem_map
                           
stem_map = {} #common_full_words(map(word_tokenize,[s for s in itertools.chain.from_iterable(map(sent_tokenize,[r.review for r in reviews]))]))

nameof = lambda wnWord: wnWord.name.split('.')[0]

# Divide the reviews by product.
def review_by_product(reviews):
    product_reviews = {}
    for r in reviews:
      if len(r.review)>0:
        try:
            product_reviews[r.product].append(r)
        except KeyError:
            product_reviews[r.product]=[r]
    return product_reviews

fst = lambda x : x[0]
def member(n): return lambda x: (n in x)
def overlap(s1): return lambda s2: float(len(s1.intersection(s2))/float(max(len(s1),len(s2))))
def pluralize(words):
    wordset = set(words)
    for w in words:
        if w[-1]=='s':
            wordset.add(w[:-1])
        else:
            wordset.add(w+'s')
    return list(wordset)

nonfeatures = pluralize(nonfeatures)

def destem(w):
    try:
        return stem_map[w]
    except:
        return w
    return w

def kbest(list1,k):
    if len(list1)<=k:
        try:
            return (zip(*list1)[1],[])
        except:
            print (list1)
        return ([],[])
    q = []
    bestK = []
    for elem in list1:
        heapq.heappush(q,elem)
    for i in range(k):
        elem = heapq.heappop(q)
        bestK.append(elem[1])

    return (bestK,zip(*q)[1])

# Return all words that are tagged as one of the provided parts of speech in a tagged sentence.
def all_POS(sentence,parts):
    return [w for (w,pos) in sentence if (pos[:2] in parts) and w not in stopwords]

# Return true if word is most likely a adjective.
def pos_adj(word):
        #pos0 = tagger.tag([word])
        pos1 = tagger.tag(['it','is', word])
        pos2 = tagger.tag(['it','is', 'a',word,'item'])
        return  (pos1[-1]==(word,'JJ')) and (pos2[-2]==(word,'JJ')) #and (pos0[-1]==(word,'JJ')) 

# Given a word, search for synonyms and related words that share similar meanings.
def construct_synset(word,POS):
    word_syns = wn.synsets(word)
    syns = [word]
    for syn in word_syns:
        if syn.lexname.split('.')[0]==POS: # and word.find(nameof(syn))>=0:
            syns = syns+[l for l in syn.lemma_names if l not in syns and l!=word]
            syns = syns+[l for l in map(nameof,syn.similar_tos()) if l not in syns and l!=word]
    if word in nonsynonyms:
        map(syns.remove, [w for w in nonsynonyms[word] if w in syns])
    allwords = synonyms_dict.keys()
    synonyms_dict[word] = set(syns)
    for syn in syns:
        if syn in allwords:
            synonyms_dict[syn].update(syns)
            synonyms_dict[word].update(syns)
    changed_synonyms.update(syns)
        
    return synonyms_dict[word]

# Keep track of the sets of available synonyms.
def update_synonyms(synonyms):
    rejected = set()
    for syn in synonyms:
        try:
            synonyms_dict[syn].update(synonyms)
        except:
            synonyms_dict[syn] = synonyms
            
# If two sets are very similar, combine them. 
def merge_sets(synonym_sets,similarity_cutoff):    
    for syndex in range(len(synonym_sets)-1):
      if len(synonym_sets[syndex])>1:  
        similarities = map(overlap(synonym_sets[syndex]),synonym_sets[syndex+1:])       
        for sdex in range(len(similarities)):#syndex+1,len(synonym_sets)):
            if similarities[sdex]>=similarity_cutoff:
                synonym_sets[syndex].update(synonym_sets[syndex+sdex+1])
                synonym_sets[syndex+sdex+1] = set()
                update_synonyms(synonym_sets[syndex])
                changed_synonyms.update(synonym_sets[syndex])
    for i in range(len(synonym_sets)-1,0,-1):
        if len(synonym_sets[i])==0:
            del synonym_sets[i]
    return synonym_sets                   

# Find the most common synonym in a given set.
def most_common_syn(synset,prop_counts,k):
    ranked_syns = []
    for syn in synset:
        heapq.heappush(ranked_syns,(-prop_counts[st.stem(syn)][0],syn))
    return set(map(fst,ranked_syns[:k+1]))

# Combine sets based on a shared most-common synonym. 
def merge_common_sets(synonym_sets,property_counts,k):
    bestK = []
    for synset in synonym_sets:
        bestK.append(most_common_syn(synset,property_counts,k))
    for syndex in range(len(synonym_sets)-1):
      if len(synonym_sets[syndex])>1:  
        similarities = map(overlap(bestK[syndex]),bestK[syndex+1:])
        for sdex in range(len(similarities)):#syndex+1,len(synonym_sets)):
            if similarities[sdex]>0:
                synonym_sets[syndex].update(synonym_sets[syndex+sdex+1])
                synonym_sets[syndex+sdex+1] = set()
    for i in range(len(synonym_sets)-1,0,-1):
        if len(synonym_sets[i])<=0:
            del synonym_sets[i]
    return synonym_sets                   

# When a synonym does not sufficiently match the set it is a part of, remove it.
def validate_synonyms(synonyms, similarity_cutoff):
    new_synonyms = []
    synonym_set = set(synonyms)
    for syn in synonyms:
        if syn in synonyms_dict.keys():
            syn_set=synonyms_dict[syn]
        else:
            syn_set=set(construct_synset(syn,'adj'))
        if len(synonym_set.intersection(syn_set))==len(synonym_set) and len(syn_set)==len(synonym_set):
            return synonyms
        if float(len(synonym_set.intersection(syn_set)))/float(len(synonyms))>=similarity_cutoff:
            new_synonyms.append(syn)
    for syn in synonyms:
        if syn in new_synonyms:
            synonyms_dict[syn].update(new_synonyms)
            changed_synonyms.update(syn)
        if syn not in new_synonyms:
            synonyms_dict[syn].difference_update(new_synonyms)
            changed_synonyms.update(syn)
    return new_synonyms

def de_stem(stemmap):
    return lambda x: stemmap[x] if x in stemmap else x
        
# Consolidate similar descriptive words into a single phrase.
def consolidate_features(feature_count,feature_properties,num_reviews,stem_map):
    similarity_cutoff = 0.2
    debug_mode = 0
    for (f,v) in feature_count.items():
        synonym_sets = []
        try:
            properties = set(map(de_stem(stem_map),feature_properties[f].keys()))
        except:
            properties = None
        while properties is not None and len(properties)>0:
            prop = properties.pop()
            has_syns = prop in synonyms_dict.keys()
            if pos_adj(prop) or has_syns:
                if has_syns:
                    syn_set = properties.intersection( synonyms_dict[prop])
                else:
                    syn_set = properties.intersection(set(construct_synset(prop,'adj')))
                properties.difference_update(syn_set)
                syn_set.add(prop)
                #already_grouped = map(overlap(syn_set),synonym_sets)
                #if len(already_grouped)>0 and max(already_grouped)>=similarity_cutoff:
                #    synonym_sets[already_grouped.index(max(already_grouped))].update(syn_set)
                #else:
                if len(syn_set)>0:
                    synonym_sets.append(syn_set)
            else:
                synonym_sets.append(set([prop]))
            
        #synonym_sets=merge_common_sets(synonym_sets,feature_properties[f],2)
        #print [s for s in synonym_sets if 'bad' in s]
        synonym_sets=merge_sets(synonym_sets,similarity_cutoff)
        final_set = set()
        for syns in synonym_sets:
            #syns = validate_synonyms(synset,similarity_cutoff)
            (bestK,restK) = kbest(zip([-feature_properties[f][st.stem(key)][0] for key in syns],syns), 2)
            new_feature = '/'.join(list(bestK)).upper()+'/'+'/'.join(list(restK)) if len(restK)>0 else '/'.join(list(bestK)).upper()
            if len(syns)>1:
                feature_properties[f][new_feature] = [0,0,0]
                for syn in syns:
                    try:
                        feature_properties[f][new_feature] = map(sum,zip(feature_properties[f][new_feature],feature_properties[f][st.stem(syn)]))
                    except:
                        if debug_mode:
                            print (f,new_feature,st.stem(syn))
                    final_set.add(st.stem(syn))
            else:
                try:
                    feature_properties[f][new_feature] = feature_properties[f][st.stem(new_feature)]
                except:
                    if debug_mode:
                        print 'problem adding new feature'
                        print (f,new_feature,st.stem(new_feature))
                final_set.add(st.stem(new_feature))
        for syn in final_set:
            if syn in feature_properties[f].keys():
                del feature_properties[f][syn]
    return feature_properties

# If we altered the synonyms, we should check that the words are sufficiently similar.
def prune_synonyms(similarity_threshold):
    while len(changed_synonyms)<0:
        word = changed_synonyms.pop()
        if len(word)<2 or word in stopwords:
            del synonyms_dict[word]
        else:
            v_set = validate_synonyms(synonyms_dict[word],similarity_threshold)
            if word in nonsynonyms:
                v_set.difference_update(nonsynonyms[word])
            diff_set = synonyms_dict[word].difference(v_set)
            for syn in v_set:
                synonyms_dict[syn]=v_set
            for syn in diff_set:
                synonyms_dict[syn].update(diff_set)
            changed_synonyms.difference_update(diff_set.union(v_set))
    return synonyms_dict

def purge_misspellings():
    if 'great' in synonyms_dict and 'bad' in synonyms_dict['great']:
        synonyms_dict['great'].remove('bad')
        if 'bad' in synonyms_dict and 'great' in synonyms_dict['bad']:
            synonyms_dict['bad'].remove('great')
    for word in synonyms_dict.keys():
        if len(synonyms_dict[word])<=2:
            del synonyms_dict[word]
    

# Remove infrequent features.
def prune_features(feature_count,feature_properties,num_reviews,stem_map):
    for (f,v) in feature_count.items():
        if v<float(num_reviews)/100.0:
            del feature_properties[f]
    return consolidate_features(feature_count,feature_properties,num_reviews,stem_map)

# Given a parsed sentence and two parts of a phrase, return the relevant keywords
def return_feature_properties(POS_sentence,i,POS1,POS2,this_review_features,mod_window):
    feature_properties = []
    use_all_mod = mod_window>0
    # Find a feature in the form of POS1 + POS2 
    if (POS_sentence[i][1][:2]==POS1 or POS1=='') and POS_sentence[i][0] not in stopwords and len(POS_sentence[i][0])>1:
        if not use_all_mod:
            all_mod = [closest_adj(POS_sentence,i)] if closest_adj(POS_sentence,i)!='' else []
        else:
            all_mod = all_POS(POS_sentence[max(0,i-mod_window):i]+POS_sentence[i+1:min(len(POS_sentence)-1,i+mod_window)],POS2)
        try:
            feature_stem = st.stem(POS_sentence[i][0])
            all_mod = map(st.stem,all_mod)
        except:
            feature_stem = POS_sentence[i][0]
            all_mod = []
        for mod in list(set(all_mod)):
            if (mod,feature_stem) not in this_review_features:
                feature_properties.append(mod)
    return feature_properties

# Parse all reviews according to parts of speech, looking for common features and their descriptive terms.
# Ex: 'the taste is good' -> good taste
# reviews: the set of reviews (the rv object)
# negate_window: how many words away is considered a negation
# adj_window: how many words away are we looking for descriptive words
# k_top_features: if we have found features, then let's only look for them. If it is empty, we need to find them. 
def pos_indicators(reviews,negate_window,adj_window,k_top_features,stem_map):
    feature_properties = {}
    feature_count = {}
    use_all_adj = True
    negating = (negate_window>0)
    rating_threshold=3.5
    total_words = 0
    total_negated =0
    feature_stems = map(st.stem,k_top_features)
    for r in range(len(reviews)):
        review = reviews[r].review
        rating = reviews[r].rating
        reviews[r].taggedPOS = map(tagger.tag,[word_tokenize(t) for t in sent_tokenize(review)])
        this_review_features = []
        
        for pos_sentence in reviews[r].taggedPOS:
            #pos_sentence = tagger.tag(re.split('\W+',sentence))
            negation_found = [0]*len(pos_sentence)
            (words,parts) = zip(*pos_sentence)
            # Find out whether there is a negation in the N words surrounding this one.
            # If there is, we will count the feature property as a negative value. 
            if negate_window>0:
                for i in range(len(words)):
                    for j in range(max(i-negate_window,0),min(i+negate_window,len(words))):
                        negation_found[i]=1 if words[j] in notwords or negation_found[i] else 0
            for i in range(len(pos_sentence)):
              if k_top_features==[] or pos_sentence[i][0] in k_top_features:
                negate_term = 1-2*negating*negation_found[i]
                # Find a feature in the form of noun+adjective or verb+adverb/adjective
                all_mod = return_feature_properties(pos_sentence,i,'NN',['NN','JJ'],this_review_features,adj_window)
                all_mod = all_mod+return_feature_properties(pos_sentence,i,'VB',['RB', 'JJ'],this_review_features,adj_window)
                try:
                    feature_stem = st.stem(pos_sentence[i][0])
                except:
                    feature_stem = pos_sentence[i][0]
                if feature_stem in feature_stems:
                    all_mod = all_mod+return_feature_properties(pos_sentence,i,'',['NN', 'JJ'],this_review_features,adj_window)
                if len(all_mod)>0:
                    try:
                        feature_count[feature_stem] = map(sum,zip([1,negate_term*(rating > rating_threshold),negate_term*(rating < rating_threshold)],feature_count[feature_stem]))
                        #feature_count[feature_stem][rating] += negate_term
                    except KeyError:
                        feature_count[feature_stem]=[1,negate_term*(rating > rating_threshold),negate_term*(rating < rating_threshold)]

                for mod in list(set(all_mod)):                    
                    if (mod,feature_stem) not in this_review_features:
                        this_review_features.append((mod,feature_stem))
                        try:
                            feature_properties[feature_stem][mod]=map(sum,zip([1,negate_term*(rating > rating_threshold),negate_term*(rating < rating_threshold)],feature_properties[feature_stem][mod]))
                        except KeyError:
                            try:
                                feature_properties[feature_stem][mod]=[1,negate_term*(rating > rating_threshold),negate_term*(rating < rating_threshold)]
                            except KeyError:
                                feature_properties[feature_stem]  = {}
                                feature_properties[feature_stem][mod]=[1,negate_term*(rating > rating_threshold),negate_term*(rating < rating_threshold)]
    
    feature_properties = prune_features(feature_count,feature_properties,len(reviews),stem_map)
    return (feature_count,feature_properties)

def print_features(feature_properties,feature_count,threshold,product,stem_map):
    print "Feature counts for product: " + product
    for f in feature_count.keys():
        if f in feature_properties.keys():
            print 'Feature ' + de_stem(stem_map)(f) + ' appears ' + str(feature_count[f][0])
            for (prop,c) in feature_properties[f].items():
                if c[0]>=threshold:
                    print de_stem(stem_map)(prop) + ': ' + str(c[0])
            
# Print most common negative and positive phrases.
def print_most_common(pos_queue,neg_queue,n,most_common_features,top_features):
    print 'Most common positive phrases:'
    i = 0
    while i<n and pos_queue!=[]:
        (priority,phrase) = heapq.heappop(pos_queue)
        if  map(phrase.find,most_common_features).count(-1)>=len(most_common_features) \
        and map(phrase.find,top_features).count(-1)<len(top_features):
            print phrase #+ ' = ' + str(-priority)
            i+=1
    print ''
    print '----------------------------------------'
    print ''
    
    print 'Most common negative phrases:'
    i=0
    while i<n and neg_queue!=[]:
        (priority,phrase) = heapq.heappop(neg_queue)
        if map(phrase.find,most_common_features).count(-1)>=len(most_common_features) \
        and map(phrase.find,top_features).count(-1)<len(top_features):
            print phrase #+ ' = ' + str(-priority)
            i+=1
    return 0

# What are the most commonly used features in both the positive and negative reviews?
def top_features(posi_ratio, nega_ratio,most_common_features,k):
    top_posi = set()
    top_nega = set()
    i = k
    while i>0 and len(posi_ratio)>0:
        (p,phrase) = heapq.heappop(posi_ratio)
        f = phrase.split()
        if  map(phrase.find,most_common_features).count(-1)>=len(most_common_features):
            top_posi.add(f[1].replace(':',''))
            i-=1
    i=k
    while i>0 and len(posi_ratio)>0:
        (p,phrase) = heapq.heappop(nega_ratio)
        f = phrase.split()
        if  map(phrase.find,most_common_features).count(-1)>=len(most_common_features):
            top_nega.add(f[1].replace(':',''))
            i-=1
    return list(top_nega.intersection(top_posi))[:4]

# Given the count of each feature and descriptive property, return a value for its
#  relative degree of positive/negative sentiment, as determined by likelihood
def pos_neg_ratio(feature_properties,feature_count,rating_count,category,stem_map):
    use_product_name = False
    pos_ll = {}
    posi_queue = []
    nega_queue = []
    numPos = rating_count[3]+rating_count[4]
    numNeg = rating_count[0]+rating_count[1]+rating_count[2]
    numTotal = sum(rating_count)
    destem = de_stem(stem_map)

    for (feature,property_count) in feature_properties.items():
        for (prop,count) in property_count.items():
            p_ratio = (float(count[1])+.01)/float(count[2]+count[1]+.02)
            n_ratio =  (float(count[2])+.01)/float(count[2]+count[1]+.02)
            p_ll = (float(count[1])+.0001)/(numPos+1*(numPos==0))
            n_ll = (float(count[2])+.0001)/(numNeg+1*(numNeg==0))
            priority = (n_ll-p_ll)#*feature_count[feature][0]
            try:
                pos_ll[feature][prop] = (log(p_ll),log(n_ll))
            except:
                pos_ll[feature] = {}
                pos_ll[feature][prop] = (log(p_ll),log(n_ll))

            heapq.heappush(posi_queue, (priority,prop + ' ' + destem(feature)+ ': ' + str(count[1]) +'/'+str(numPos+1*(numPos==0))+ ': ' + str(count[2]) +'/'+str(numNeg+1*(numNeg==0))))
            heapq.heappush(nega_queue,(-priority,prop + ' ' + destem(feature)+ ': ' + str(count[2]) +'/'+str(numNeg+1*(numNeg==0))+ ': ' + str(count[1]) +'/'+str(numPos+1*(numPos==0))))
                
    #print_most_common(posi_queue[:],nega_queue[:],20,category,feature_count.keys())
    return (posi_queue,nega_queue,pos_ll)

# For each feature/property combination, find the pos/negative ratio. 
def compute_feature_weights(feature_properties,k_top_features):
    weights = {}
    for f in k_top_features:
        weights[f] = {}
        for (prop_merged,count) in feature_properties[st.stem(f)].items():
            weight = float(count[1])/max(count[1],float(count[1]+count[2]+100*((count[1]+count[2])<=0)))
            weight =max(weight,1-float(count[2])/max(count[0],float(100*((count[0])<=0))))
            for prop in prop_merged.split('/'):
                weights[f][prop] = min(5,weight*4+1)
                #if weights[f][prop]<2:
                #    print (f,prop,weights[f][prop],count,prop_merged,float(count[1]+count[2]+100*((count[1]+count[2])<=0)),100*((count[1]+count[2])<=0))
    return weights

# For each feature/property pair, find this product's value given the counts and relative weightings.
def product_feature_values(feature_properties,features,weights,threshold):
    feature_values = {}
    feature_counts = {}
    for f in features:
        feature = st.stem(f)
        if feature in feature_properties.keys():
            feature_values[f] = 0
            feature_counts[f] = 0
            for (prop, count) in feature_properties[feature].items():
                f_property = prop.split('/')[0]
                for c in range(1,len(count)):
                    try:
                        feature_values[f]+=(weights[f][f_property]*count[c])
                    except KeyError:
                        feature_values[f]+=0
                if f_property in weights[f]:
                    feature_counts[f] += sum(count[1:])
            if feature_counts[f]>threshold:
                feature_values[f] = (max(0,float(feature_values[f])),float(feature_counts[f]))
            else:
                feature_values[f] = (-1,0)
        else:
            feature_values[f] = (-1,0)
    return feature_values

# Given a single review and the weights and existing feature values for the product, update those feature values.
def update_one_review(review,weights,feature_values,negate_window,adj_window,k_top_features,stem_map):
    (feature_count,feature_properties)=pos_indicators([review],negate_window,adj_window,k_top_features,stem_map)
    destem = de_stem(stem_map)
    for (feature,property_count) in feature_properties.items():
        for (prop,prop_count) in property_count:
            single_prop = prop.split('/')[0]
            (feature,(value,count)) = feature_values[feature]
            feature_values[feature] = (value+prop_count*weights[destem(feature)][single_prop],count+prop_count)
    return feature_values

def print_feature_values(feature_values,numReviews,avg_rating):
    for (prod,prod_features) in feature_values.items():
        print 'The values for ' + prod + ' that has ' + str(numReviews[prod]) + ' reviews'
        print 'Average rating: ' + "{0:.1f}".format(avg_rating[prod])
        for (f,(value,count)) in prod_features.items():
            if value>0:
                print f + ' : ' + "{0:.1f}".format(float(value)/float(count))
            else:
                print f + ' not found'

def print_category_features(weights):
    for (cat,cat_weights) in weights.items():
        print 'Features for ' + cat + ': ' + str(cat_weights.keys())

# Calling this function will output the values for the most common features of the given category.
# Ex: tvs, shampoo, etc. Needs to be in a csv file. If using some other data source,
# use a different extract_reviews method. 
def extracted_features(categoryName):
    print_results = True
    product_cutoff = 3  # How many reviews do we need to make a proper classification?
    negation_window = 4 # How close is a negation word ('not', 'never') to make us not count the feature? (number of words)
    adjacency_window= 5 # How close should a descriptive term be to get counted? (number of words)
    similarity_threshold = 0.25 # How similar should words be to consider them synonyms?
    category_sample_size = 5000 # This should be enough to extract the main features and build weights.
    maxNumReviews = 50000    # Make this number bigger if there are more reviews to consider
    inputFilename = 'reviews_'+categoryName+'.csv'

    all_reviews = extract_reviews(inputFilename,maxNumReviews)
    ratings = [r.rating for r in all_reviews]
    product_reviews = review_by_product(all_reviews)
    stem_map = common_full_words(map(word_tokenize,[s for s in itertools.chain.from_iterable(map(sent_tokenize,[r.review for r in all_reviews]))]))

    ratings_count = [ratings.count(1),ratings.count(2),ratings.count(3),ratings.count(4),ratings.count(5)]
    
    # If we have already discovered good enough weights for a feature set for this product category,
    # we can save some work. 
    try:     
        f = open('weights.pickle')
        product_weights = pickle.load(f)
        f.close()
        if len(product_weights)==0:
            product_weights = {}
    except:
        product_weights = {}

    # Uncomment this if the category weights are sufficient and we want to redo each product.
    if False : #or categoryName in product_weights: 
        weights = product_weights[categoryName]
        k_top_features = weights.keys()
    else:
        # Parse all reviews (or a large sample of them) according to parts of speech, looking for common features and their descriptive terms.
        random.shuffle(all_reviews)
        (all_feature_count,all_feature_properties) = pos_indicators(all_reviews[:category_sample_size],negation_window,adjacency_window,[],stem_map)
        (all_posi_q,all_nega_q,all_pn_likelihood)=pos_neg_ratio(all_feature_properties,all_feature_count,ratings_count,nonfeatures+pluralize(categoryName.split('-')),stem_map)
        k_top_features = top_features(all_posi_q[:],all_nega_q[:],nonfeatures+pluralize(categoryName.split('-')),20)

        # Print most common negative and positive phrases.
        if print_results:
            print_most_common(all_posi_q[:],all_nega_q[:],5,nonfeatures+categoryName.split('-'),k_top_features)

        # For each feature/property combination, find the pos/negative ratio. 
        weights = compute_feature_weights(all_feature_properties,k_top_features)
        product_weights[categoryName] = weights
        try:
            f=open('weights.pickle', "wb" )
            pickle.dump(product_weights, f)
            f.close()
        except:
            categoryName = categoryName

    product_feature_properties = {}
    product_feature_count = {}
    feature_values = {}
        
    numReviews = {}
    avg_rating = {}
    
    # Do the same for each product, and record how it stacks up to the category. 
    for prod in product_reviews.keys():
      if len(product_reviews[prod])>product_cutoff:
        if prod in product_feature_properties.keys():
            feature_properties = product_feature_properties[prod]
            feature_count = product_feature_count[prod]
        else:
            (feature_count,feature_properties) = pos_indicators(product_reviews[prod],negation_window,adjacency_window,k_top_features,stem_map)
            product_feature_properties[prod]=feature_properties
            product_feature_count[prod]=feature_count
        numReviews[prod] = len(product_reviews[prod])
        feature_values[prod] = product_feature_values(feature_properties,k_top_features,weights,product_cutoff)
        avg_rating[prod] = sum([r.rating for r in product_reviews[prod]])/float(len(product_reviews[prod]))
    if print_results:
        print_feature_values(feature_values, numReviews, avg_rating)
        print_category_features(product_weights)

    if len(changed_synonyms)>0:
        if print_results:
            print str(len(changed_synonyms)) +  ' Number of synonym updates!!!'
            print changed_synonyms
        prune_synonyms(similarity_threshold)
        #if len(changed_synonyms)*10>len(synonyms_dict):
        #    purge_misspellings()
        f=open('synonyms.pickle', "wb" )
        pickle.dump(synonyms_dict, f)
        f.close()
    return (feature_values,product_feature_properties,weights,avg_rating)

# Perform the analysis
(feature_values,product_feature_properties,weights,avg_rating)=extracted_features(categoryName);

# Correlation between ratings and feature values
def correlate_features(feature_values,avg_rating):
    avgs = np.array([0.0]*len(avg_rating.keys()))
    f_values = {}
    features = feature_values[avg_rating.keys()[0]].keys()
    for f in features:
        f_values[f] = np.array([0.0]*len(avgs))

    i=0
    for (prod, avg) in avg_rating.items():
        avgs[i]=avg
        zeroFound = False
        for (f,(value,count)) in feature_values[prod].items():
            f_values[f][i]=float(value)/float(count)
            zeroFound=zeroFound or (value<0)
        if zeroFound:
            avgs[i]=0
            for f in features:
                f_values[f][i]=0
        i+=1
                    
    for i in range(len(features)):
        print str(np.corrcoef([v for v in f_values[features[i]] if v>0],[v for v in avgs if v>0])) +'is correlation for ' + features[i]
        for j in range(i+1,len(features)):
            print str(np.corrcoef([v for v in f_values[features[i]] if v>0],[v for v in f_values[features[j]] if v>0])) +'is correlation for ' + features[i] + ' and ' + features[j]
    return f_values

correlate_features(feature_values,avg_rating)
