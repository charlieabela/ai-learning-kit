"""ICS5111 Session 02 (graph embeddings): inspectable CPU teaching helpers, not a benchmark library.

Used in the three Session 02 notebooks as T = kit.tools() after kit.start("ICS5111", "S02").
Cora comes from its owner (LINQS) through ai_learning_kit.datasets, checked against the
checksum of the archive the reference results were computed with. The karate club and Les
Miserables come with networkx. reference/ holds the frozen results of the run used in the
slides (one split, seed 5111).

Ported from the companion-ZIP helpers, version 2026-10-27, without changing any computation;
adr_graph() was left out because no Session 02 notebook uses it.
"""
from pathlib import Path
import json, time, tarfile
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.special import expit
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from .... import datasets

ROOT = Path(__file__).parent          # ROOT / 'reference' holds the frozen reference results
SEED = 5111
VERSION = '2026-10-27'   # revised Session 02 helpers; earlier copies lack karate(), lesmis() and the other additions

def toy_graph():
    return nx.Graph([(1,2),(1,3),(2,3),(2,4),(4,5),(5,6),(5,7),(6,7)])

def draw_toy(ax=None, highlight=()):
    if ax is None: _,ax=plt.subplots(figsize=(7,3))
    g=toy_graph();pos={1:(0,1),3:(0,-1),2:(1,0),4:(2,0),5:(3,0),6:(4,1),7:(4,-1)}
    nx.draw(g,pos,ax=ax,with_labels=True,node_color='#4279B7',font_color='white',node_size=850,width=2,edge_color='#9CA9B6')
    if highlight: nx.draw_networkx_edges(g,pos,edgelist=list(zip(highlight[:-1],highlight[1:])),ax=ax,edge_color='#D37B35',width=4)
    ax.margins(.15);return ax

def transition(g, previous, current, p=1., q=1., weighted=False):
    """Second-order node2vec transition, for an undirected graph, p,q > 0."""
    if g.is_directed(): raise ValueError('This teaching implementation expects an undirected graph.')
    if p<=0 or q<=0: raise ValueError('p and q must be positive')
    neighbours=sorted(g.neighbors(current))
    weights=[]
    for x in neighbours:
        bias=1. if previous is None else (1/p if x==previous else (1. if g.has_edge(previous,x) else 1/q))
        weights.append(bias*(g[current][x].get('weight',1.) if weighted else 1.))
    w=np.asarray(weights,dtype=float)
    return neighbours, w/w.sum() if len(w) else w

def walks(g, n_walks=3, length=16, p=1., q=1., seed=SEED, weighted=False):
    """Stops at isolated nodes. Caches transitions; can be costly on high-degree graphs."""
    rng=np.random.default_rng(seed); nodes=sorted(g); result=[];cache={}
    for _ in range(n_walks):
        for start in rng.permutation(nodes):
            w=[start.item() if isinstance(start, np.generic) else start]   # plain Python value, not a numpy scalar
            for _ in range(length-1):
                key=(w[-2] if len(w)>1 else None,w[-1])
                if key not in cache: cache[key]=transition(g,*key,p=p,q=q,weighted=weighted)
                ns,pr=cache[key]
                if not ns:break
                w.append(ns[int(rng.choice(len(ns),p=pr))])
            result.append(w)
    return result

def context_pairs(walk_list, window=2):
    if window<1: raise ValueError('window must be >= 1')
    return np.asarray([(w[i],w[j]) for w in walk_list for i in range(len(w))
                       for j in range(max(0,i-window),min(len(w),i+window+1)) if i!=j],dtype=np.int64).reshape(-1,2)

def fit_sgns(pairs,n_nodes,dim=16,negatives=3,epochs=2,lr=.15,batch=256,seed=SEED):
    """Fixed-window skip-gram negative sampling. Returns source Z and context C.
    Repeated visits may create self-node pairs at different positions. Noise is
    context frequency ** .75, sampled with replacement; accidental positives are
    retained as noise (not asserted to be nonedges). Batch summed SGD gradients
    approximate successive updates. This is an explicit teaching variant.
    """
    if len(pairs)==0: raise ValueError('No pairs: inspect isolates or walk length.')
    rng=np.random.default_rng(seed)
    z=rng.normal(0,.1,(n_nodes,dim));c=rng.normal(0,.1,(n_nodes,dim))
    counts=np.bincount(pairs[:,1],minlength=n_nodes).astype(float)**.75
    noise=counts/counts.sum();history=[];t0=time.perf_counter()
    for epoch in range(epochs):
        total=0.
        for start in range(0,len(pairs),batch):
            if start==0: order=rng.permutation(len(pairs))
            pp=pairs[order[start:start+batch]];u=pp[:,0]
            targets=np.column_stack([pp[:,1],rng.choice(n_nodes,(len(pp),negatives),p=noise)])
            zz=z[u].copy();cc=c[targets].copy()
            scores=np.einsum('bd,bkd->bk',zz,cc)
            labels=np.zeros(scores.shape);labels[:,0]=1
            total+=np.sum(np.logaddexp(0,scores)-labels*scores)
            err=expit(scores)-labels
            # The sum objective is stated in the notes; scale rate to reduce collisions.
            np.add.at(z,u,-lr*np.einsum('bk,bkd->bd',err,cc))
            np.add.at(c,targets.reshape(-1),(-lr*err[:,:,None]*zz[:,None,:]).reshape(-1,dim))
        history.append(float(total/len(pairs)))
    return z,c,{'loss_per_positive_pair':history,'seconds':time.perf_counter()-t0,'pairs':len(pairs)}

def embed(g,**kwargs):
    nodes=sorted(g);mapping={n:i for i,n in enumerate(nodes)};h=nx.relabel_nodes(g,mapping)
    wk={k:kwargs.pop(k) for k in list(kwargs) if k in ['n_walks','length','p','q','weighted']}
    seed=kwargs.get('seed',SEED);t0=time.perf_counter()
    ws=walks(h,seed=seed,**wk);pairs=context_pairs(ws,kwargs.pop('window',2))
    z,c,info=fit_sgns(pairs,len(nodes),**kwargs)
    info['total_seconds']=time.perf_counter()-t0
    return nodes,z,info

def load_cora():
    with tarfile.open(datasets.fetch('cora', quiet=True)) as t:
        rows=[line.decode().strip().split() for line in t.extractfile('cora/cora.content')]
        citations=[line.decode().strip().split() for line in t.extractfile('cora/cora.cites')]
    rows=sorted(rows,key=lambda x:x[0]);ids=np.array([r[0] for r in rows]);lookup={v:i for i,v in enumerate(ids)}
    x=sparse.csr_matrix(np.array([r[1:-1] for r in rows],dtype=np.float64))
    classes,y=np.unique([r[-1] for r in rows],return_inverse=True)
    g=nx.Graph();g.add_nodes_from(range(len(ids)))
    g.add_edges_from((lookup[a],lookup[b]) for a,b in citations if a!=b)
    train,rest=train_test_split(np.arange(len(ids)),train_size=.2,stratify=y,random_state=SEED)
    val,test=train_test_split(rest,train_size=.25,stratify=y[rest],random_state=SEED)
    split={'train':np.sort(train),'val':np.sort(val),'test':np.sort(test)}
    return g,x,y,ids,classes,split

def classify(features,y,split,test=False):
    """Select regularisation with validation labels only. Test evaluation is explicit."""
    candidates=[]
    for strength in [.1,1.,10.]:
        model=make_pipeline(StandardScaler(with_mean=False),LogisticRegression(C=strength,max_iter=1500,random_state=SEED))
        model.fit(features[split['train']],y[split['train']])
        candidates.append((accuracy_score(y[split['val']],model.predict(features[split['val']])),strength,model))
    score,strength,model=max(candidates,key=lambda t:t[0]);out={'C':strength,'validation_accuracy':float(score)}
    if test:
        pred=model.predict(features[split['test']]);truth=y[split['test']]
        out.update(test_accuracy=float(accuracy_score(truth,pred)),test_macro_f1=float(f1_score(truth,pred,average='macro')))
    return out,model

def structural_features(g):
    nodes=sorted(g);pr=nx.pagerank(g);cl=nx.clustering(g)
    return np.array([[g.degree(n),cl[n],pr[n]] for n in nodes])

def split_links(g,seed=SEED):
    """Exclude held-out positives before feature computation; negatives disjoint."""
    rng=np.random.default_rng(seed);edges=np.array(sorted(tuple(sorted(e)) for e in g.edges()))
    rng.shuffle(edges);n=len(edges);nv=max(1,int(.1*n));nt=nv
    positive={'val':edges[:nv],'test':edges[nv:nv+nt],'train':edges[nv+nt:]}
    known=set(map(tuple,edges));negative=[];seen=set();nodes=sorted(g)
    while len(negative)<n:
        a,b=sorted(rng.choice(nodes,2,replace=False));pair=(int(a),int(b))
        if pair not in known and pair not in seen:negative.append(pair);seen.add(pair)
    negative=np.array(negative);negative={'val':negative[:nv],'test':negative[nv:nv+nt],'train':negative[nv+nt:]}
    train_graph=g.copy();train_graph.remove_edges_from(np.vstack([positive['val'],positive['test']]))
    assert all(not train_graph.has_edge(*e) for e in np.vstack([positive['val'],positive['test']]))
    return train_graph,positive,negative

def common_neighbour_auc(train_graph,positive,negative):
    scores=[len(set(train_graph[a])&set(train_graph[b])) for a,b in np.vstack([positive,negative])]
    return float(roc_auc_score([1]*len(positive)+[0]*len(negative),scores))

def save_handover(g,ids,split,z,config,directory='s02_handover'):
    out=Path(directory);out.mkdir(exist_ok=True)
    np.savez_compressed(out/'cora-s02-baseline.npz',paper_ids=ids,edges=np.array(sorted(g.edges())),embeddings=z,**split)
    config=dict(config);config['cora_sha256']=datasets.sha256_of(datasets.fetch('cora', quiet=True))
    config['split_seed']=SEED;config['protocol']='20/20/60 stratified; full undirected topology; transductive node classification'
    (out/'experiment.json').write_text(json.dumps(config,indent=2))
    return out

# ---------------------------------------------------------------------------------------------
# Additions for the revised Session 02 (karate, Les Miserables, walk spread, inductive features,
# link features, TransE). The functions above are unchanged, so earlier reference results hold.
# ---------------------------------------------------------------------------------------------
def karate():
    """Zachary's karate club (networkx copy) and the faction of each member: 0 = Mr. Hi, 1 = Officer."""
    g=nx.karate_club_graph()
    club=np.array([g.nodes[n]['club']=='Officer' for n in sorted(g)]).astype(int)
    return g,club

def lesmis():
    """Les Miserables co-appearance network (Knuth, 1993), nodes relabelled 0..76; names in attribute 'name'."""
    return nx.convert_node_labels_to_integers(nx.les_miserables_graph(),label_attribute='name')

def sgns_snapshots(pairs,n_nodes,dim=2,negatives=5,epochs=30,lr=.025,batch=64,seed=SEED):
    """Same algorithm as fit_sgns, but returns a copy of Z after every epoch (snapshot 0 = random start)
    and the loss per positive pair for every epoch. Used to watch a small embedding train."""
    rng=np.random.default_rng(seed)
    z=rng.normal(0,.1,(n_nodes,dim));c=rng.normal(0,.1,(n_nodes,dim))
    counts=np.bincount(pairs[:,1],minlength=n_nodes).astype(float)**.75
    noise=counts/counts.sum();snaps=[z.copy()];losses=[]
    for epoch in range(epochs):
        order=rng.permutation(len(pairs));total=0.
        for start in range(0,len(pairs),batch):
            pp=pairs[order[start:start+batch]];u=pp[:,0]
            targets=np.column_stack([pp[:,1],rng.choice(n_nodes,(len(pp),negatives),p=noise)])
            zz=z[u].copy();cc=c[targets].copy()
            scores=np.einsum('bd,bkd->bk',zz,cc)
            labels=np.zeros(scores.shape);labels[:,0]=1
            total+=np.sum(np.logaddexp(0,scores)-labels*scores)
            err=expit(scores)-labels
            np.add.at(z,u,-lr*np.einsum('bk,bkd->bd',err,cc))
            np.add.at(c,targets.reshape(-1),(-lr*err[:,:,None]*zz[:,None,:]).reshape(-1,dim))
        losses.append(float(total/len(pairs)));snaps.append(z.copy())
    return snaps,losses

def walk_spread(g,walk_list):
    """Mean hop distance between the first and last node of each walk, and mean number of distinct nodes."""
    far=[nx.shortest_path_length(g,w[0],w[-1]) for w in walk_list]
    return float(np.mean(far)),float(np.mean([len(set(w)) for w in walk_list]))

def neighbour_mean_features(g,x,visible):
    """Inductive features without training: [own features | mean features of the visible neighbours].
    visible: boolean array; edges to invisible nodes are ignored (a new paper can cite only papers that
    already exist). Nodes without a visible neighbour get zeros in the second block."""
    a=nx.to_scipy_sparse_array(g,nodelist=range(len(g)),format='csr')
    a=a.multiply(np.asarray(visible,dtype=float)[None,:]).tocsr()
    deg=np.asarray(a.sum(1)).ravel()
    mean=sparse.diags(1/np.maximum(deg,1))@a@x
    return sparse.hstack([x,mean]).tocsr(),deg

def edge_features(z,pairs):
    """Hadamard (element-wise) product of the two node vectors: one feature vector per node pair."""
    pairs=np.asarray(pairs);return z[pairs[:,0]]*z[pairs[:,1]]

def fit_transe(triples,n_ent,n_rel,dim=2,epochs=800,lr=.01,margin=.5,seed=SEED):
    """Minimal TransE (Bordes et al., 2013): train h + r close to t for true triples, with a margin
    ranking loss against triples whose head or tail is replaced at random. Entity vectors are kept
    inside the unit ball. A teaching sketch for a handful of triples, not a library implementation."""
    rng=np.random.default_rng(seed)
    E=rng.normal(0,1,(n_ent,dim));E/=np.linalg.norm(E,axis=1,keepdims=True)
    R=rng.normal(0,.1,(n_rel,dim));tr=np.asarray(triples)
    for _ in range(epochs):
        for h,r,t in tr[rng.permutation(len(tr))]:
            if rng.random()<.5: h2,t2=h,int(rng.integers(n_ent))
            else: h2,t2=int(rng.integers(n_ent)),t
            dp=E[h]+R[r]-E[t];dn=E[h2]+R[r]-E[t2]
            if margin+np.linalg.norm(dp)-np.linalg.norm(dn)>0:
                gp=dp/(np.linalg.norm(dp)+1e-9);gn=dn/(np.linalg.norm(dn)+1e-9)
                E[h]-=lr*gp;E[t]+=lr*gp;R[r]-=lr*gp
                E[h2]+=lr*gn;E[t2]-=lr*gn;R[r]+=lr*gn
        E/=np.maximum(1,np.linalg.norm(E,axis=1,keepdims=True))
    return E,R
