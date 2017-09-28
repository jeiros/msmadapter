from msmadapter.adaptive import App, Adaptive
from msmadapter.plot_utils import plot_spawns
from msmadapter.traj_utils import traj_from_stateinds, write_tleap_script
app = App(meta='meta.pandas.pickl')
ad = Adaptive(app=app, stride=50)
ad.run()
