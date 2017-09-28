from msmadapter.adaptive import App, Adaptive
app = App(meta='meta.pandas.pickl')
ad = Adaptive(app=app, stride=2)
ad.fit_model()
frames = ad.find_respawn_conformations()
