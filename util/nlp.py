from __future__ import division, print_function

import sys
import os
import numpy as np
import gensim
import nltk
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
sys.path.append('.')
import util
import string
# os.environ['GLOG_minloglevel'] = "4"
# import caffe

class NLP_Tools():
    def __init__(self, opts = {}):
        self.w2v = None
        self.qa = None
        if 'gpu_id' not in opts:
            opts['gpu_id'] = 0
        self.opts = opts
        self.wnl = WordNetLemmatizer()
        stopwords = nltk.corpus.stopwords.words('english')
        self.stopwords = stopwords[0:stopwords.index('of')] + stopwords[(stopwords.index('under')+1)::]\
                            + [p for p in string.punctuation]

    def _load(self, model):
        if model == 'word2vec' and self.w2v == None:
            print('loading word2vec model')
            self.w2v = gensim.models.word2vec.Word2Vec.load_word2vec_format(\
                        'external/GoogleNews-vectors-negative300.bin.gz', binary = True)
            # self.w2v = gensim.models.word2vec.Word2Vec.load_word2vec_format(\
            #             'external/mini_word2vec_300.bin.gz', binary = True)
        elif model == 'qa' and self.qa == None:
            print('loading qa embedding model')
            sys.path.append('modules/question_answerer')
            import question_answerer
            self.qa = question_answerer.Question_Answerer(opts = {'gpu_id': self.opts['gpu_id']})


    def dist(self, v1, v2, metric = 'euclidean'):
        if metric == 'euclidean':
            return np.sqrt(np.dot(v1 - v2, v1 - v2))
        elif metric == 'cos':
            # return 1 - np.dot(v1/v1.dot(v1), v2/v2.dot(v2))
            return (1 - v1.dot(v2)/np.sqrt(v1.dot(v1) * v2.dot(v2)))/2

        else:
            return -1

    def w2v_embd_std(self, pos_text, neg_text = None):
        self._load('word2vec')
        w2v = self.w2v
        vecp = [w2v[w] for w in nltk.word_tokenize(pos_text.lower()) if w not in self.stopwords and w in w2v.vocab]
        vecp = np.zeros(w2v.vector_size, dtype = np.float32) if not vecp else np.array(vecp).sum(axis = 0)
        if neg_text:
            vecn = [w2v[w] for w in nltk.word_tokenize(neg_text.lower()) if w not in self.stopwords and w in w2v.vocab]
            vecn = np.zeros(w2v.vector_size, dtype = np.float32) if not vecn else np.array(vecn).sum(axis = 0)
            vecp -= vecn
        return gensim.matutils.unitvec(vecp)

    def w2v_embd(self, pos_text, neg_text = None):
        if not self.w2v:
            self._load('word2vec')
        w2v = self.w2v
        vecp = [w2v[w] for w in nltk.word_tokenize(pos_text.lower()) if w in w2v.vocab]
        vecp = np.zeros(w2v.vector_size, dtype = np.float32) if not vecp else np.array(vecp).sum(axis = 0)
        if neg_text:
            vecn = [w2v[w] for w in nltk.word_tokenize(neg_text.lower()) if w in w2v.vocab]
            vecn = np.zeros(w2v.vector_size, dtype = np.float32) if not vecn else np.array(vecn).sum(axis = 0)
            vecp -= vecn
        return gensim.matutils.unitvec(vecp)

    def w2v_dist(self, s1, s2):
        # self._load('word2vec')
        
        v1 = self.w2v_embd(s1)
        v2 = self.w2v_embd(s2)
        return (1 - v1.dot(v2))/2
        # return self.dist(v1, v2, metric = 'cos')

    def w2v_dist_std(self, s1, s2):
        v1 = self.w2v_embd_std(s1)
        v2 = self.w2v_embd_std(s2)
        return (1 - v1.dot(v2))/2

    def qa_embd(self, pos_text):
        self._load('qa')
        return self.qa.compute_question_embedding(pos_text)

    def qa_dist(self, s1, s2):
        v1 = self.qa_embd(s1)
        v2 = self.qa_embd(s2)
        return self.dist(v1, v2)


    def select_different_question(self, ref, cand, num):
        n_cand = len(cand)
        embd_ref = [self.qa_embd(q) for q in ref]
        embd_cand = [self.qa_embd(q) for q in cand]
        assert(n_cand >= num)
        if num == 0:
            return [],[]

        slct_idx = []
        distance = []
        dist_cand = np.ones(n_cand, dtype = np.float32) * 1e8
        for i, v_r in enumerate(embd_ref):
            for j, v_c in enumerate(embd_cand):
                dist_cand[j] = min(dist_cand[j], self.dist(v_r, v_c))

        n  = 0
        while(n < num):
            k = np.argmax(dist_cand)
            slct_idx.append(k)
            distance.append(dist_cand[k])
            dist_cand[k] = -1
            for i, d in enumerate(dist_cand):
                if d > 0:
                    dist_cand[i] = min(d, self.dist(embd_cand[i], embd_cand[k]))
            n += 1
        return slct_idx, distance

    def question_type(self, question):
        '''
        0: what (object)
        1: QTYPE_COLOR
        2: where
        3: how many / what number
        4: how
        5: who
        6: QTYPE_TIME
        7: why
        8: what (action)
        9: what (substance)
        10: how old
        11: QTYPE_SHAPE
        '''

        question = question.lower()
        if question.startswith('what'):
            if 'color ' in question or 'colors ' in question or question.endswith('color'):
                return 1, 'QTYPE_COLOR'
            elif question.startswith('what time') or 'the time' in question:
                return 6, 'QTYPE_TIME'
            elif question.startswith('what number of') or 'the number of ' in question:
                return 3, 'QTYPE_QUANTITY'
            elif 'do' in question or 'doing' in question and not (question.startswith('what do')):
                return 8, 'QTYPE_DO'
            elif question.endswith('made of'):
                return 9, 'QTYPE_SUBSTANCE'
            elif question.startswith('what shape'):
                return 11, 'QTYPE_SHAPE'
            else:
                return 0, 'QTYPE_ENTITY'
        elif question.startswith('where'):
            return 2, 'QTYPE_POSITION'
        elif question.startswith('how'):
            if question.startswith('how many'):
                return 3, 'QTYPE_QUANTITY'
            elif question.startswith('how old'):
                return 10, 'QTYPE_AGE'
            else:
                return 4, 'QTYPE_HOW'
        elif question.startswith('who'):
            return 5, 'QTYPE_WHO'
        elif question.startswith('when'):
            return 6, 'QTYPE_TIME'
        elif question.startswith('why'):
            return 7, 'QTYPE_WHY'
        else:
            return -1, 'QTYPE_UNKOWN'

    def question_headword(self, question):
        # q_pos = nltk.pos_tag(nltk.word_tokenize(question.lower()))
        q_token = [w for w in nltk.word_tokenize(question.lower())]
        q_pos = nltk.pos_tag(q_token)
        q_pos = [(self._lemmatize(w),p[0:2]) for w, p in q_pos]
        q_type = self.question_type(question)[1]

        # print(q_pos)
        if ('be', 'VB') in q_pos:
            q_pos.remove(('be', 'VB'))

        try:
            idx = q_pos.index(('\'s', 'PO'))
            if q_pos[idx-1][1] == 'NN' and q_pos[idx+1][1] == 'NN':
                q_pos.pop(idx)
                q_pos.pop(idx-1)
        except:
            pass




        if q_type == 'QTYPE_TIME' and ('time', 'NN') in q_pos:
            q_pos.remove(('time', 'NN'))
        elif q_type == 'QTYPE_COLOR' and ('color', 'NN') in q_pos:
            q_pos.remove(('color', 'NN'))
            
        elif q_type == 'QTYPE_QUANTITY' and ('number', 'NN') in q_pos:
            q_pos.remove(('number', 'NN'))
        elif q_type == 'QTYPE_ENTITY':
            if ('kind', 'NN') in q_pos:
                q_pos.remove(('kind', 'NN'))
            if ('type', 'NN')in q_pos:
                q_pos.remove(('type', 'NN'))
        elif q_type == 'QTYPE_SHAPE' and ('shape', 'NN') in q_pos:
            q_pos.remove(('shape', 'NN'))

        

        first_n = None
        first_v = None

        for idx, (w, pos) in enumerate(q_pos):
            if pos.startswith('NN'):
                if idx < len(q_pos) - 1 and q_pos[idx+1][0] == '\'s':
                    continue
                else:
                    first_n = w
                    break

        for w, pos in q_pos:
            if pos.startswith('VB'):
                first_v = w
                break


        if q_type in {'QTYPE_TIME', 'QTYPE_WHY', 'QTYPE_HOW'}:
            return None, None
        if q_type not in {'QTYPE_ENTITY', 'QTYPE_DO', 'QTYPE_WHO'} and first_n:
            first_v = None
        if q_type in {'QTYPE_WHO'}:
            first_n = None

        return first_n, first_v
        


    def question_match_w2v(self, q1, q2):
        q1_type = self.question_type(q1)[0]
        q2_type = self.question_type(q2)[0]
        if q1_type != q2_type:
            return 0
        else:
            q1_hn, q1_hv = self.question_headword(q1)
            q2_hn, q2_hv = self.question_headword(q2)

            # if q1_hn and q2_hn:
            #     if q1_hv and q2_hv:
            #         return 1 - (self.w2v_dist(q1_hn, q2_hn) + \
            #                     self.w2v_dist(q1_hv, q2_hv) + \
            #                     self.w2v_dist(q1_rest, q2_rest))/3
            #     else:
            #         return 1 - (self.w2v_dist(q1_hn, q2_hn) + \
            #             self.w2v_dist(q1_rest, q2_rest))/2
            # else:
            #     return 1 - self.w2v_dist(q1_rest, q2_rest)

            if q1_hn and q2_hn:
                s_hn = 1 - self.w2v_dist(q1_hn, q2_hn)
            else:
                s_hn = 1

            if q1_hv and q2_hv:
                s_hv = 1 - self.w2v_dist(q1_hv, q2_hv)
            else:
                s_hv = 1

            s_q = 1 - self.w2v_dist_std(q1, q2)
            return s_hn * s_hv * s_q


    def _lemmatize(self, w):
        w1 = self.wnl.lemmatize(w, wordnet.NOUN)
        if w1 == w:
            w1 = self.wnl.lemmatize(w1, wordnet.VERB)
        return w1

def _compute_pn(idx, output_list, hypo, refs, scores, ngram):
    '''
    helper function for delta_bleu
    output_list[idx]: tuple (len_h, len_r, d_UP1,...,d_UPn, d_DOWN1,...,d_DOWNn)
    idx: index
    hypo: generated sentenc
    refs: references
    scores: score of references, in range [-1, 1]
    n_gram: max length of gram
    '''

    from nltk.util import ngrams

    # hypo = nltk.word_tokenize(hypo.lower())
    # refs = [nltk.word_tokenize(ref.lower()) for ref in refs]
    # hypo = hypo.lower().split()
    # refs = [ref.lower().split() for ref in refs]

    if not isinstance(hypo, list):
        hypo = hypo.lower().split()
    refs = [ref.lower().split() if not isinstance(ref, list) else ref for ref in refs]

    up = [0] * ngram
    down = [0] * ngram
    max_score = max(scores)
    for n in xrange(1, ngram+1):
        hypo_ngram = list(ngrams(hypo, n))
        refs_ngram = [list(ngrams(ref, n)) for ref in refs]

        for g in set(hypo_ngram):
            c = hypo_ngram.count(g)
            down[n-1] += c * max_score
            match = [min(c, ref.count(g)) * s for s, ref in zip(scores, refs_ngram)]
            match = [m for m in match if m != 0]
            if not match:
                match = [0]
            up[n-1] += max(match) 

    len_h = len(hypo)
    len_rs = np.array([len(ref) for ref in refs])
    len_r = len_rs[np.argmin(np.abs(len_rs-len_h))]

    output_list[idx] = tuple([len_h, len_r] + up + down)
    # print('Index: %d'%idx)

def delta_bleu1(hypos, refs_corpus, scores_corpus, ngram):
    '''
    hypos[i], str, the i-th generated sentence
    refs_corpus[i], list(str), the references of i-th sentence
    scores_corpus[i], list(float), the scores of references in i-th reference set
    ngram: max lengh of gram
    '''

    small = 1e-8
    tiny = 1e-15


    num = len(hypos)
    assert(len(refs_corpus) == num)
    assert(len(scores_corpus) == num)
    output_list = [None] * num
    for idx, (hypo, refs, scores) in enumerate(zip(hypos, refs_corpus, scores_corpus)):
        _compute_pn(idx, output_list, hypo, refs, scores, ngram)
    output = zip(*output_list)

    # output[0]: list of len_h
    # output[1]: list of len_r
    # output[1+n]: list of pn_up
    # output[1+ngram+n]: list of pn_down
    assert(len(output) == ngram*2 + 2)
    assert(len(output[0]) == num)
    sum_len_h = sum(output[0])
    sum_len_r = sum(output[1])
    BP = 1.0 if sum_len_h > sum_len_r else np.exp(1 - sum_len_r/sum_len_h)
    pn = [(sum(output[1+n])+tiny)/(sum(output[1+ngram+n])+small) + small for n in xrange(1, ngram+1)]
    d_bleu = np.zeros(ngram, dtype = np.float32)
    for n in xrange(ngram):
        d_bleu[n] = BP * np.exp(np.mean(np.log(pn[0:n+1])))

    # util.io.save_json(output, 'temp/bleu_data.json')
    # print(d_bleu)

    return d_bleu

def delta_bleu(hypos, refs_corpus, scores_corpus, ngram):
    '''
    hypos[i], str, the i-th generated sentence
    refs_corpus[i], list(str), the references of i-th sentence
    scores_corpus[i], list(float), the scores of references in i-th reference set
    ngram: max lengh of gram
    '''
    from nltk.util import ngrams
    from scipy.stats.mstats import gmean
    small = 1e-8
    tiny = 1e-15

    num = len(hypos)
    assert(len(refs_corpus) == num)
    assert(len(scores_corpus) == num)

    up = np.zeros(ngram, dtype = np.float)
    down = np.zeros(ngram, dtype = np.float)
    len_r = 0
    len_h = 0
    for idx, (hypo, refs, scores) in enumerate(zip(hypos, refs_corpus, scores_corpus)):
        # match ngram for one sample
        hypo = hypo if isinstance(hypo, list) else hypo.lower().split()
        refs = [ref if isinstance(ref, list) else ref.lower().split() for ref in refs]
        for n in xrange(1, ngram + 1):
            max_score = max(scores)
            hypo_ngram = list(ngrams(hypo, n))
            refs_ngram = [list(ngrams(ref, n)) for ref in refs]

            for g in set(hypo_ngram):
                c = hypo_ngram.count(g)
                down[n-1] += max_score * c
                match = [min(c, ref.count(g)) * s for s, ref in zip(scores, refs_ngram) if g in ref]
                if match:
                    up[n-1] += max(match)
        len_h += len(hypo)
        len_rs = np.array([len(ref) for ref in refs])
        len_r += len_rs[np.argmin(np.abs(len_rs - len(hypo)))]


    up = np.maximum(up, tiny)
    pn = (up) / (down+small)
    BP = 1.0 if len_h > len_r else np.exp(1 - len_r/len_h)
    d_bleu = np.zeros(ngram, dtype = np.float)
    for n in xrange(0,ngram):
        d_bleu[n] = gmean(pn[0:n+1])

    return BP * d_bleu

def automatic_metrics(hypos, refs_corpus, scores_corpus, ngram):
    pass

def baidu_translate(sent_list):
    import md5
    import urllib
    import httplib

    if not isinstance(sent_list, list):
        sent_list = [sent_list]

    appid = '20170220000039498'
    secretKey = 'xXw6phjoGtYYZ6uzV19G'

    httpClient = None
    myurl = '/api/trans/vip/translate'
    fromLang = 'en'
    toLang = 'zh'

    q = ('\n'.join(sent_list)).encode('utf-8')
    salt = np.random.randint(32768, 65536)
    sign = appid+q+str(salt)+secretKey
    m1 = md5.new()
    m1.update(sign)
    sign = m1.hexdigest()
    myurl = myurl+'?appid='+appid+'&q='+urllib.quote(q)+'&from='+fromLang+'&to='+toLang+'&salt='+str(salt)+'&sign='+sign
     
    try:
        httpClient = httplib.HTTPConnection('api.fanyi.baidu.com')
        httpClient.request('GET', myurl)

        # response is a HTTPResponse object
        response = httpClient.getresponse()
        # print(response.read())
        # output_list[idx] = response.read()
        output = response.read()
    except Exception, e:
        print(e)
    finally:
        if httpClient:
            httpClient.close()
    return output
