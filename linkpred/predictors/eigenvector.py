import networkx as nx

from ..evaluation import Scoresheet
from ..network import rooted_pagerank, simrank, nonlocal_pagerank
from ..util import progressbar
from .base import Predictor
import numpy as np
import scipy as sp

class RootedPageRank(Predictor):
    def predict(self, nbunch=None, alpha=0.85, beta=0, weight="weight", k=None):
        """Predict using rooted PageRank.

        Parameters
        ----------

        nbunch : iterable collection of nodes, optional
            node(s) to calculate PR for (default: all)

        alpha : float, optional
            PageRank probability that we will advance to a neighbour of the
            current node in a random walk

        beta : float, optional
            Normally, we return to the root node with probability 1 - alpha.
            With this parameter, we can also advance to a random other node in
            the network with probability beta. Thus, we get back to the root
            node with probability 1 - alpha - beta. This is off (0) by default.

        weight : string or None, optional
            The edge attribute that holds the numerical value used for
            the edge weight.  If None then treat as unweighted.

        k : int or None, optional
            If `k` is `None`, this predictor is applied to the entire network.
            If `k` is an int, the predictor is applied to a subgraph consisting
            of the k-neighbourhood of the current node.
            Results are often very similar but much faster.

        See documentation for linkpred.network.rooted_pagerank for these
        parameters.

        """
        res = Scoresheet()
        if nbunch is None:
            nbunch = self.G.nodes()
        for u in progressbar(nbunch):
            if not self.eligible_node(u):
                continue
            if k is None:
                G = self.G
            else:
                # Restrict to the k-neighbourhood subgraph
                G = nx.ego_graph(self.G, u, radius=k)
            pagerank_scores = rooted_pagerank(G, u, alpha, beta, weight)
            for v, w in pagerank_scores.items():
                if w > 0 and u != v and self.eligible_node(v):
                    res[(u, v)] += w
        return res


class SimRank(Predictor):
    def predict(self, c=0.8, num_iterations=10, weight="weight"):
        r"""Predict using SimRank

        .. math ::
            sim(u, v) = \frac{c}{|N(u)| \cdot |N(v)|} \sum_{p \in N(u)}
                        \sum_{q \in N(v)} sim(p, q)

        where `N(v)` is the set of neighbours of node `v`.

        Parameters
        ----------
        c : float, optional
            decay factor, determines how quickly similarity decreases

        num_iterations : int, optional
            number of iterations to calculate

        weight: string or None, optional
            If None, all edge weights are considered equal.
            Otherwise holds the name of the edge attribute used as weight.

        """
        res = Scoresheet()
        nodelist = list(self.G.nodes)
        sim = simrank(self.G, nodelist, c, num_iterations, weight)
        (m, n) = sim.shape
        assert m == n

        for i in range(m):
            # sim(a, b) = sim(b, a), leading to a 'mirrored' matrix.
            # We start the column range at i + 1, such that we only look at the
            # upper triangle in the matrix, excluding the diagonal:
            # sim(a, a) = 1.
            u = nodelist[i]
            for j in range(i + 1, n):
                if sim[i, j] > 0:
                    v = nodelist[j]
                    if self.eligible(u, v):
                        res[(u, v)] = sim[i, j]
        return res

class NonLocalPageRank(Predictor):
    def predict(self, nbunch=None, alpha=0.85, beta=0, weight="weight", k=None, type="power", gamma=1):
        """Predict using rooted PageRank.

        Parameters
        ----------

        nbunch : iterable collection of nodes, optional
            node(s) to calculate PR for (default: all)

        alpha : float, optional
            PageRank probability that we will advance to a neighbour of the
            current node in a random walk

        beta : float, optional
            Normally, we return to the root node with probability 1 - alpha.
            With this parameter, we can also advance to a random other node in
            the network with probability beta. Thus, we get back to the root
            node with probability 1 - alpha - beta. This is off (0) by default.

        weight : string or None, optional
            The edge attribute that holds the numerical value used for
            the edge weight.  If None then treat as unweighted.

        k : int or None, optional
            If `k` is `None`, this predictor is applied to the entire network.
            If `k` is an int, the predictor is applied to a subgraph consisting
            of the k-neighbourhood of the current node.
            Results are often very similar but much faster.

        type : string
            If type is "power" the adjacency matrix is transformed by using the
            function f(x) = 1/x^gamma, if type is "exponential" the adjacency matrix
            is transformed by using the exponential function f(x) = exp(-gamma x)

        gamma : parameter value for the transformation

        See documentation for linkpred.network.rooted_pagerank for these
        parameters.

        """
        res = Scoresheet()
        if nbunch is None:
            nbunch = self.G.nodes()
        for u in progressbar(nbunch):
            if not self.eligible_node(u):
                continue
            if k is None:
                G = self.G
                W = np.copy(self.DistanceMatrix)
            else:
                # Restrict to the k-neighbourhood subgraph
                G = nx.ego_graph(self.G, u, radius=k)
                W = nx.floyd_warshall_numpy(G)

            # To make the call faster we compute just one time the distance
            # matrix
            P = np.asarray(W)
            if type == "power":
                P[P != 0] = 1.0/np.power(P[P != 0],gamma)
            else:
                P[P != 0] = np.exp(-gamma*P[P != 0])
        
            S = sp.array(P.sum(axis=1)).flatten()
            S[S != 0] = 1.0 / S[S != 0]
            Q = sp.sparse.spdiags(S.T, 0, *P.shape, format='csr')
            P = Q * P
            
            pagerank_scores = nonlocal_pagerank(G, P, S, u, alpha, beta, weight, type, gamma)

            for v, w in pagerank_scores.items():
                if w > 0 and u != v and self.eligible_node(v):
                    res[(u, v)] += w
        return res
