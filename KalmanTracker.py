import numpy as np
from filterpy.kalman import KalmanFilter

# (x, y, z, w, l, h, angle, v_x, v_y, v_z, v_angle)
class KalmanTracker():
    def __init__(self, box, id, type):
        self.id = id
        self.type = type
        
        self.kf = KalmanFilter(dim_x=11, dim_z=7)  

        self.kf.F = np.array([[1,0,0,0,0,0,0,1,0,0,0],   
                              [0,1,0,0,0,0,0,0,1,0,0],
                              [0,0,1,0,0,0,0,0,0,1,0],
                              [0,0,0,1,0,0,0,0,0,0,0],  
                              [0,0,0,0,1,0,0,0,0,0,0],
                              [0,0,0,0,0,1,0,0,0,0,0],
                              [0,0,0,0,0,0,1,0,0,0,1],
                              [0,0,0,0,0,0,0,1,0,0,0],
                              [0,0,0,0,0,0,0,0,1,0,0],
                              [0,0,0,0,0,0,0,0,0,1,0],
                              [0,0,0,0,0,0,0,0,0,0,1]])  

        self.kf.H = np.array([[1,0,0,0,0,0,0,0,0,0,0], 
                              [0,1,0,0,0,0,0,0,0,0,0],
                              [0,0,1,0,0,0,0,0,0,0,0],
                              [0,0,0,1,0,0,0,0,0,0,0],
                              [0,0,0,0,1,0,0,0,0,0,0],
                              [0,0,0,0,0,1,0,0,0,0,0],
                              [0,0,0,0,0,0,1,0,0,0,0]])

        self.kf.P[7:,7:] *= 1000. 
        self.kf.P *= 10.

        self.kf.Q[7:,7:] *= 0.01

        self.kf.x[:7] = box.reshape((7, 1))
    
    def update(self, box):
        self.kf.update(box)
        
    def predict(self):
        self.kf.predict()
        
    def get_state(self):
        return self.kf.x[:7].reshape((7,))
