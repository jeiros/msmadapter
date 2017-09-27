from msmadapter.Simulation import App, Adaptive
app = App(meta='meta.pandas.pickl')
ad = Adaptive(app=app, stride=20)
ad.fit_model()
ad.find_respawn_fame()
