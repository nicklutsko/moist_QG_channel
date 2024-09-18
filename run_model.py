import numpy as np
import moist_2LQG

Ls = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
for i in range( 2, 9 ):
    model = moist_2LQG.MoistQGModel(L=Ls[i], filename = "data/moist_QG_run" + str(i + 1) + ".nc")
    model.run_simulation()