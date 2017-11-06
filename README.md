# MSMadapter

Implement an adaptive search for MD simulations run with AMBER.

Use your own msmbuilder defined models for the search.


# Example
```python 
from msmadapter.adaptive import App, Adaptive
from msmadapter.model_utils import retrieve_clusterer
from msmadapter.plot_utils import plot_tica_landscape, plot_clusters, plot_spawns
from matplotlib import pyplot as pp
app = App(meta='WT_meta.pickl', project_name='adaptive', user_HPC='username', ngpus=8)
ad = Adaptive(app=app, stride=1)
ad.fit_model()
spawns_cluster = ad.respawn_from_clusterer()
spawns_tica = ad.respawn_from_tICs()
spawns_MSM = ad.respawn_from_MSM()
clusterer = retrieve_clusterer(ad.model)
ax = plot_tica_landscape(ad.ttrajs)
plot_clusters(clusterer)
plot_spawns(spawns_cluster, ad.ttrajs)
plot_spawns(spawns_tica, ad.ttrajs, color='green')
plot_spawns(spawns_MSM, ad.ttrajs, color='purple')
f = pp.gcf()
f.savefig('spawns.pdf')
```