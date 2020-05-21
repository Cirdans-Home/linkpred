#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 17:50:19 2020

@author: cirdan
"""

import linkpred
import random
from matplotlib import pyplot as plt

random.seed(100)

# Read network
G = linkpred.read_network('examples/inf1990-2004.net')

# Create test network
test = G.subgraph(random.sample(G.nodes(), 300))

# Exclude test network from learning phase
training = G.copy()
training.remove_edges_from(test.edges())

# Local
rootpr = linkpred.predictors.RootedPageRank(training, excluded=training.edges())
rootpr_results = rootpr.predict()

test_set = set(linkpred.evaluation.Pair(u, v) for u, v in test.edges())
evaluationpr = linkpred.evaluation.EvaluationSheet(rootpr_results, test_set)



# NonLocal
rootnonlocal = linkpred.predictors.NonLocalPageRank(training, excluded=training.edges())
rootnonlocal_results = rootnonlocal.predict()

test_set = set(linkpred.evaluation.Pair(u, v) for u, v in test.edges())
evaluation_nonlocal = linkpred.evaluation.EvaluationSheet(rootnonlocal_results, test_set)

fig, ax = plt.subplots()
ax.plot(evaluationpr.recall(), evaluationpr.precision(),label='PageRank')
ax.plot(evaluation_nonlocal.recall(), evaluation_nonlocal.precision(),label='NonLocal PageRank')
ax.axis('equal')
leg = ax.legend(loc='upper right', frameon=False);
fig