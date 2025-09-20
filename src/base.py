import numpy as np
from utils import Geometrics_info, Displacement_calcu
from scipy.spatial.distance import cdist

class faults():
    '''
    '''
    def __init__(self, cvalue, Fau_km, MaxMw):
        '''
        '''
        self.cvalue = cvalue
        self.Fau_km = Fau_km
        self.MaxMw = np.array(MaxMw).reshape(-1, 1) #MwD
        self.get_faultparas()
        self.calc_rms()

    def get_faultparas(self):
        '''
        '''
        # 找到FE_deg最后一列唯一值
        line_e = np.unique(self.Fau_km[:, -1])
        line_nums_e = []
        for item in line_e:
            first_idx = np.where(self.Fau_km[:, -1] == item)[0][0]
            last_idx  = np.where(self.Fau_km[:, -1] == item)[0][-1] + 1
            line_nums_e.append([first_idx, last_idx, item])
        line_nums_e = np.array(line_nums_e, dtype=int)
        #
        n = 0
        ell = np.empty((0, 4))
        Aspect3D, Magrecord = np.empty((0, 2)), np.empty((0, 2))
        FaultParameters = np.empty((0, 30))
        for i in range(len(line_e)):
            n += 1

            subfau_km = self.Fau_km[line_nums_e[i, 0]:line_nums_e[i, 1], :]
            m1 = np.mean(subfau_km[:, 0:3], axis=0)
            (a, b, c, p1_u, p1_d, p2_u, p2_d, p3_u, p3_d, az, dip, aspect3d, ellipsoid_points) = Geometrics_info(m1, subfau_km)
            ell = np.vstack(( ell, np.hstack((ellipsoid_points, np.full((ellipsoid_points.shape[0], 1), n))) ))  

            # 更新 Magrecord
            mag_col = subfau_km[:, 3].reshape(-1, 1)
            mag_tmp = np.hstack(( mag_col, np.full((mag_col.shape[0], 1), n) ))
            Magrecord = np.vstack((Magrecord, mag_tmp)) # mag n

            # 更新 FaultParameters
            median_xyz = np.median(subfau_km[:, 0:3], axis=0)
            sp_len = subfau_km.shape[0]
            sp_col14 = subfau_km[0, 13]  # MATLAB 第14列，Python是13

            FaultParameters = np.vstack(( FaultParameters, 
                np.hstack((median_xyz, a, b, c, p1_u, p1_d, p2_u, p2_d, p3_u, p3_d, az, dip, aspect3d, self.cvalue, sp_len, sp_col14)) 
            ))

            # 更新 ASPECT3D
            Aspect3D = np.vstack((Aspect3D, np.array([[sp_len, aspect3d]])))
        self.ell = ell
        self.FaultParameters = FaultParameters
        self.Aspect3D = Aspect3D
        self.Magrecord = Magrecord

    def calc_rms(self):
        '''
        '''
        
        SubRLength, SubRWidth  = [], []
        for i in range(self.FaultParameters.shape[0]):
            dist1 = cdist(self.FaultParameters[i, 6:9].reshape(1, -1), self.FaultParameters[i, 9:12].reshape(1, -1))[0, 0]
            dist2 = cdist(self.FaultParameters[i, 12:15].reshape(1, -1), self.FaultParameters[i, 15:18].reshape(1, -1))[0, 0]
            SubRLength.append(dist1)
            SubRWidth.append(dist2)
        SubRLength, SubRWidth = np.array(SubRLength).reshape(-1,1), np.array(SubRWidth).reshape(-1,1)  # (n,1)

        d = Displacement_calcu(self.MaxMw) # (n,1)
        # 计算理论矩
        self.Motheo = 3 * 1e10 * (1e3 * SubRLength) * (1e3 * SubRWidth) * (1e3 * SubRLength) * d
        Moreal = []
        for _i in range(1, int(self.Magrecord[-1, 1]) + 1):
            num2 = np.where(self.Magrecord[:, 1] == _i)[0]
            Moreal.append(np.sum(10 ** (1.5 * self.Magrecord[num2, 0] + 9.1)))
        self.Moreal = np.array(Moreal).reshape(-1,1)
        self.rsm = np.log10(self.Moreal) / np.log10(self.Motheo)