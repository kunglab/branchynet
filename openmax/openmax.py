import scipy.spatial.distance as spd
import scipy as sp
import numpy as np
import libmr

def compute_distance(query_channel, channel, mean_vec, distance_type = 'eucos'):
    """ Compute the specified distance type between chanels of mean vector and query image.
    In caffe library, FC8 layer consists of 10 channels. Here, we compute distance
    of distance of each channel (from query image) with respective channel of
    Mean Activation Vector. In the paper, we considered a hybrid distance eucos which
    combines euclidean and cosine distance for bouding open space. Alternatively,
    other distances such as euclidean or cosine can also be used. 
    
    Input:
    --------
    query_channel: Particular FC8 channel of query image
    channel: channel number under consideration
    mean_vec: mean activation vector

    Output:
    --------
    query_distance : Distance between respective channels

    """

    if distance_type == 'eucos':
        query_distance = spd.euclidean(mean_vec[channel, :], query_channel)/200. + spd.cosine(mean_vec[channel, :], query_channel)
    elif distance_type == 'euclidean':
        query_distance = spd.euclidean(mean_vec[channel, :], query_channel)/200.
    elif distance_type == 'cosine':
        query_distance = spd.cosine(mean_vec[channel, :], query_channel)
    else:
        print "distance type not known: enter either of eucos, euclidean or cosine"
    return query_distance

def compute_channel_distances(mean_train_channel_vector, features, n_channels=1):
    """
    Input:
    ---------
    mean_train_channel_vector : mean activation vector for a given class. 
                                It can be computed using MAV_Compute.py file
    features: features for the category under consideration
    Output:
    ---------
    channel_distances: dict of distance distribution from MAV for each channel. 
    distances considered are eucos, cosine and euclidean
    """
    

    eucos_dist, eu_dist, cos_dist = [], [], []
    for channel in range(n_channels):
        eu_channel, cos_channel, eu_cos_channel = [], [], []
        # compute channel specific distances
        for feat in features:
            eu_channel += [spd.euclidean(mean_train_channel_vector[channel, :], feat[channel, :])/200.]
            cos_channel += [spd.cosine(mean_train_channel_vector[channel, :], feat[channel, :])]
            eu_cos_channel += [spd.euclidean(mean_train_channel_vector[channel, :], feat[channel, :])/200. +
                               spd.cosine(mean_train_channel_vector[channel, :], feat[channel, :])]
        eu_dist += [eu_channel]
        cos_dist += [cos_channel]
        eucos_dist += [eu_cos_channel]

    # convert all arrays as scipy arrays
    eucos_dist = sp.asarray(eucos_dist)
    eu_dist = sp.asarray(eu_dist)
    cos_dist = sp.asarray(cos_dist)

    # assertions for length check
    assert eucos_dist.shape[0] == 1
    assert eu_dist.shape[0] == 1
    assert cos_dist.shape[0] == 1
    assert eucos_dist.shape[1] == len(features)
    assert eu_dist.shape[1] == len(features)
    assert cos_dist.shape[1] == len(features)

    channel_distances = {'eucos': eucos_dist, 'cosine': cos_dist, 'euclidean':eu_dist}
    return channel_distances

def compute_MAV(features, correct_labels, n_classes):
    correct_features = {}
    for category in range(n_classes):
        correct_features[category] = []
    for i,feature in enumerate(features):
        try:
            predicted_category = feature.argmax()
            if predicted_category and correct_labels[i] and predicted_category == category:
                correct_features[predicted_category] += [feature]
        except TypeError:
            continue
    mean_feature_vecs={}
    for category in range(n_classes):
        mean_feature_vecs[category] = sp.mean(correct_features[category],axis=0)
    return mean_feature_vecs

def compute_distances(features, correct_labels):
    n_channels = features[0].shape[0]
    n_classes = features[0].shape[1]
    
    correct_features = {}
    for category in range(n_classes):
        correct_features[category] = []
    for i,feature in enumerate(features):
        try:
            predicted_category = feature.argmax()
            if predicted_category == correct_labels[i]:
                correct_features[predicted_category] += [feature]
        except TypeError:
            continue
    mean_feature_vecs = {}
    distance_distributions = {}
    for category in range(n_classes):
        mean_feature_vecs[category] = sp.mean(correct_features[category],axis=0)
        distance_distributions[category] = compute_channel_distances(mean_feature_vecs[category], correct_features[category], n_channels=n_channels)
    return distance_distributions,mean_feature_vecs

#    correct_features = []
#    for feature in features:
#        try:
#            predicted_category = feature.argmax()
#            if predicted_category == category:
#                correct_features += [feature]
#        except TypeError:
#            continue
#    #mean_feature_vec = sp.mean(correct_features,axis=0)
#    distance_distribution = compute_channel_distances(mean_feature_vec, correct_features)
#    return distance_distribution


def weibull_tailfitting(meantrain_vecs,
                        distance_distribution, 
                        tailsize = 20, 
                        distance_type = 'eucos',
                        n_channels = 1,
                        n_classes = 10):
    weibull_model = {}
    for category in range(n_classes):
        weibull_model[category] = {}
        distance_scores = distance_distribution[category][distance_type]
        meantrain_vec = meantrain_vecs[category]
        
        weibull_model[category]['distances_%s'%distance_type] = distance_scores
        weibull_model[category]['mean_vec'] = meantrain_vec
        weibull_model[category]['weibull_model'] = []

        for channel in range(n_channels):
            mr = libmr.MR()
            tailtofit = sorted(distance_scores[channel,:])[-tailsize:]
            mr.fit_high(tailtofit, len(tailtofit))
            weibull_model[category]['weibull_model'] += [mr]
    return weibull_model

def query_weibull(category, weibull_model, distance_type = 'eucos'):
    """ Query through dictionary for Weibull model.
    Return in the order: [mean_vec, distances, weibull_model]
    
    Input:
    ------------------------------
    category : class id
    weibull_model: dictonary of weibull models for 
    """

    category_weibull = []
    category_weibull += [weibull_model[category]['mean_vec']]
    category_weibull += [weibull_model[category]['distances_%s' %distance_type]]
    category_weibull += [weibull_model[category]['weibull_model']]
    
    return category_weibull

def computeOpenMaxProbability(openmax_fc8, openmax_score_u, n_channels=1, n_classes=10):
    """ Convert the scores in probability value using openmax
    
    Input:
    ---------------
    openmax_fc8 : modified FC8 layer from Weibull based computation
    openmax_score_u : degree
    Output:
    ---------------
    modified_scores : probability values modified using OpenMax framework,
    by incorporating degree of uncertainity/openness for a given class
    
    """
    
    prob_scores, prob_unknowns = [], []
    for channel in range(n_channels):
        channel_scores, channel_unknowns = [], []
        for category in range(n_classes):
            channel_scores += [sp.exp(openmax_fc8[channel, category])]
                    
        total_denominator = sp.sum(sp.exp(openmax_fc8[channel, :])) + sp.exp(sp.sum(openmax_score_u[channel, :]))
        prob_scores += [channel_scores/total_denominator ]
        prob_unknowns += [sp.exp(sp.sum(openmax_score_u[channel, :]))/total_denominator]
        
    prob_scores = sp.asarray(prob_scores)
    prob_unknowns = sp.asarray(prob_unknowns)

    scores = sp.mean(prob_scores, axis = 0)
    unknowns = sp.mean(prob_unknowns, axis=0)
    modified_scores =  scores.tolist() + [unknowns]
    assert len(modified_scores) == 11
    return modified_scores

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference

def recalibrate_scores(feature, weibull_model, alpharank=10, distance_type='eucos', n_channels=1, n_classes=10):
    ranked_list = feature.argsort().ravel()[::-1]
    alpha_weights = [((alpharank+1) - i)/float(alpharank) for i in range(1, alpharank+1)]
    
    ranked_alpha = sp.zeros(10)
    for i in range(len(alpha_weights)):
        ranked_alpha[ranked_list[i]] = alpha_weights[i]

    # Now recalibrate each fc8 score for each channel and for each class
    # to include probability of unknown
        
    openmax_fc8, openmax_score_u = [], []
    for channel in range(n_channels):
        channel_scores = feature[channel, :]
        openmax_fc8_channel = []
        openmax_fc8_unknown = []
        count = 0
        for categoryid in range(n_classes):
            # get distance between current channel and mean vector
            category_weibull = query_weibull(categoryid, weibull_model, distance_type = distance_type)
            channel_distance = compute_distance(channel_scores, channel, category_weibull[0],
                                                distance_type = distance_type)

            # obtain w_score for the distance and compute probability of the distance
            # being unknown wrt to mean training vector and channel distances for
            # category and channel under consideration
            wscore = category_weibull[2][channel].w_score(channel_distance)
            modified_fc8_score = channel_scores[categoryid] * ( 1 - wscore*ranked_alpha[categoryid] )
            openmax_fc8_channel += [modified_fc8_score]
            openmax_fc8_unknown += [channel_scores[categoryid] - modified_fc8_score ]

        # gather modified scores fc8 scores for each channel for the given image
        openmax_fc8 += [openmax_fc8_channel]
        openmax_score_u += [openmax_fc8_unknown]
    openmax_fc8 = sp.asarray(openmax_fc8)
    openmax_score_u = sp.asarray(openmax_score_u)
    
    # Pass the recalibrated fc8 scores for the image into openmax    
    openmax_probab = computeOpenMaxProbability(openmax_fc8, openmax_score_u, n_channels = n_channels, n_classes = n_classes)
    softmax_probab = softmax(feature.ravel())
    return sp.asarray(openmax_probab), sp.asarray(softmax_probab)

class OpenMax(object):
    def __init__(self, n_classes=10, distance_type='eucos', tailsize=20, alpharank=10, n_channels=1):
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.tailsize = tailsize
        self.distance_type = distance_type
        self.alpharank = alpharank
    def fit(self,features, labels):
        distance_distributions, meantrain_vecs = compute_distances(features, labels)
        self.weibull_model = weibull_tailfitting(meantrain_vecs, distance_distributions, tailsize=self.tailsize, distance_type=self.distance_type, n_channels = self.n_channels, n_classes = self.n_classes)
        return self
    def transform(self, feature):
        return recalibrate_scores(feature, self.weibull_model, alpharank=self.alpharank, distance_type=self.distance_type, n_channels = self.n_channels, n_classes = self.n_classes)

    
def expand_dims(features):
    return sp.expand_dims(features,axis=1)

def get_openmax_scores(features, openmax):
    openmax_vs = []
    softmax_vs = []
    for feature in features:
        openmax_v, softmax_v = openmax.transform(feature)
        openmax_vs.append(openmax_v)
        softmax_vs.append(softmax_v)
    openmax_vs = sp.array(openmax_vs)
    softmax_vs = sp.array(softmax_vs)
    softmax_vs = sp.hstack([softmax_vs, sp.zeros([softmax_vs.shape[0],1])])
    return openmax_vs, softmax_vs

def plot_openmax_mean(openmax_vs, softmax_vs, title=''):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,7))
    plt.title(title)
    openmax_mean = sp.mean(openmax_vs,0)
    softmax_mean = sp.mean(softmax_vs,0)
    om_handle, = plt.plot(openmax_mean, label='openmax')
    sm_handle, = plt.plot(softmax_mean, label='softmax')
    plt.legend(handles=[om_handle, sm_handle], loc=0)
    plt.show()
    print openmax_mean
    print softmax_mean
    return
