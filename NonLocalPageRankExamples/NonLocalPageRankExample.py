#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is an example of link-prediction by means of the NonLocal Pagerank as 
described in:
    [1] Cipolla, S., Durastante, F., Tudisco, F. NonLocal PageRank
Created on Thu May 21 17:50:19 2020

"""

import linkpred
import random
import networkx as nx
from matplotlib import pyplot as plt
from sklearn.metrics import auc
from numpy import floor, logspace, array

random.seed(100)

# Read network
G = linkpred.read_network('NonLocalPageRankExamples/USAir97.net')

# We remove self loops
G.remove_edges_from(nx.selfloop_edges(G))

# We treat unweighted networks:
for u,v,d in G.edges(data=True):
    d['weight']=1

# Create test network
test = G.subgraph(random.sample(G.nodes(), floor(0.40*G.number_of_nodes()).astype(int) ))

# Exclude test network from learning phase
training = G.copy()
training.remove_edges_from(test.edges())

## LinkPrediction with the Baseline method
baseline = linkpred.predictors.Random(training, excluded=training.edges())
baseline_results = baseline.predict()

test_set = set(linkpred.evaluation.Pair(u, v) for u, v in test.edges())
evaluationbaseline = linkpred.evaluation.EvaluationSheet(baseline_results, test_set)
aucbaseline = auc(evaluationbaseline.recall(), evaluationbaseline.precision())

## Plotting the Precision/Recall curves
fig, ax = plt.subplots(figsize=[6.4, 4.8],dpi=300)
plt.xlabel('Recall')
plt.ylabel('Precision')
ax.loglog(evaluationbaseline.recall(), evaluationbaseline.precision(),label='Baseline')

## LinkPredicition with the Local (Rooted) PageRank
rootpr = linkpred.predictors.RootedPageRank(training, excluded=training.edges())
rootpr_results = rootpr.predict()

test_set = set(linkpred.evaluation.Pair(u, v) for u, v in test.edges())
evaluationpr = linkpred.evaluation.EvaluationSheet(rootpr_results, test_set)
auclocal = auc(evaluationpr.recall(), evaluationpr.precision())

## Plotting the Precision/Recall curves
ax.plot(evaluationpr.recall(), evaluationpr.precision(),label='PageRank')

## LinkPrediction with the NonLocal (Rooted) PageRank
# We have modified the original predictors class to include a key for the 
# distance matrix, in this way we can compute it just one time for all the
# iteration. If the parameter k is passed, i.e., if we require the analysis to
# work on a neighboroud of the node, then the precomputation is unnecessary,
# because it is then done on the fly on the reduced graph. 
W = nx.floyd_warshall_numpy(training) 
index = 0
aucnonlocal = array([0.,1.,2.,3.,4.,5.])
for gamma in logspace(-2.0, 1.0, num=6, endpoint=True): # Coefficient of the NonLocality
    rootnonlocal = linkpred.predictors.NonLocalPageRank(training, DistanceMatrix=W, excluded=training.edges())
    rootnonlocal_results = rootnonlocal.predict(gamma = gamma)
    test_set = set(linkpred.evaluation.Pair(u, v) for u, v in test.edges())
    evaluation_nonlocal = linkpred.evaluation.EvaluationSheet(rootnonlocal_results, test_set)
    aucnonlocal[index] = auc(evaluation_nonlocal.recall(), evaluation_nonlocal.precision())
    index = index+1
    ax.loglog(evaluation_nonlocal.recall(), evaluation_nonlocal.precision(),label='NonLocal PageRank alpha='+str(gamma))


# Plot of the results
leg = ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., frameon=False);
fig.savefig('USAir97.eps', bbox_extra_artists=(leg,), bbox_inches='tight', format='eps')

# Save the AUC to file
outfile = open("linkpredlog.txt","a+")
outfile.write("Baseline AUC "+str(aucbaseline)+"\n")
outfile.write("PageRank AUC "+str(auclocal)+"\n")
index = 0
for gamma in logspace(-2.0, 1.0, num=6, endpoint=True):
    outfile.write("NonLocal PageRank "+str(gamma)+" AUC "+str(aucnonlocal[index])+"\n")
    index=index+1
outfile.close()