import numpy as np
from sklearn.cluster import DBSCAN
from scipy.stats import chi2
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA
def Displacement_calcu(Mw):
    # Mw: 输入标量或数组

    # 原始数据
    Mag = np.array([-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7])
    D = np.array([0.001, 0.0015, 0.002, 0.005, 0.01, 0.8, 0.85, 0.9, 1, 1, 1]) * 1e-2

    # 拟合函数模型
    def model(Mag, a, b, c, d):
        return a / (1 + np.exp(b * Mag + c)) + d

    # 初始猜测
    initial_guess = [1, 1, 1, 1]

    # 曲线拟合
    params_fit, _ = curve_fit(model, Mag, D, p0=initial_guess)

    # 拟合参数
    a, b, c, d = params_fit

    # 输出拟合后的 D（支持标量或数组 Mw）
    Mw = np.asarray(Mw)
    D_out = a / (1 + np.exp(b * Mw + c)) + d
    return D_out

def Candidate_fault_classify(PBAD_Multiple, line_ab, candi_fau, Class_Mode, main_fault, angle_tolerance):
    '''
    '''
    
    rangeup = []
    rangedown = []
    lim = []

    # 取B向量计算角度
    B = line_ab[:, 3:6]
    
    # 断层编号唯一值及区间索引
    fault_ids = np.unique(candi_fau[:, -1])
    line_nums_1 = []
    for fid in fault_ids:
        indices = np.where(candi_fau[:, -1] == fid)[0]
        line_nums_1.append([indices[0], indices[-1]])
    line_nums_1 = np.array(line_nums_1, dtype=int)

    if main_fault==None:
        length_I = 1
    else:
        length_I = len(main_fault)
        
    for i in range(length_I):
        # 走向角度转换范围0-180
        fai_rad_ori = np.arctan2(B[:, 0], B[:, 1])
        fai_deg = np.degrees(fai_rad_ori)
        fai_deg[fai_deg < 0] += 180
        fai= fai_deg.copy()

        bin_edges = np.arange(0, 190, 10)
        bin_counts, _ = np.histogram(fai_deg, bins=bin_edges)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        if Class_Mode == 1:
            idd = np.argmax(bin_counts)  # 找最大值索引
            PBAD = np.median(np.abs(fai - bin_centers[idd]))
            faiup = bin_centers[idd] + PBAD_Multiple * PBAD
            faidown = bin_centers[idd] - PBAD_Multiple * PBAD

            # 找bin_edges中小于等于bin_centers[idd]的最大索引
            nbu_arr = np.where(bin_edges <= bin_centers[idd])[0]
            nbu = nbu_arr[-1] if len(nbu_arr) > 0 else None

            # 找bin_edges中大于等于bin_centers[idd]的最小索引
            nbd_arr = np.where(bin_edges >= bin_centers[idd])[0]
            nbd = nbd_arr[0] if len(nbd_arr) > 0 else None
        elif Class_Mode == 2:
            faiup = main_fault[i] + angle_tolerance
            faidown = main_fault[i] - angle_tolerance

            # 找bin_edges中小于等于main_fault[i]的最大索引
            nbu_arr = np.where(bin_edges <= main_fault[i])[0]
            nbu = nbu_arr[-1] if len(nbu_arr) > 0 else None

            # 找bin_edges中大于等于main_fault[i]的最小索引
            nbd_arr = np.where(bin_edges >= main_fault[i])[0]
            nbd = nbd_arr[0] if len(nbd_arr) > 0 else None
        else:
            raise ValueError("Invalid fai_MODE")

        fu = faiup > 180
        fd = faidown < 0
        if fu:
            faiup -= 180
        if fd:
            faidown += 180

        if fu:
            uplim = np.min(np.where(bin_edges >= faiup)[0])
            rangeup.append([0, bin_edges[uplim]])
            rangeup.append([bin_edges[nbu], bin_edges[-1]])
        else:
            uplim = np.min(np.where(bin_edges >= faiup)[0])
            rangeup.append([bin_edges[nbu], bin_edges[uplim]])

        if fd:
            downlim = np.max(np.where(bin_edges <= faidown)[0])
            rangedown.append([0, bin_edges[nbd]])
            rangedown.append([bin_edges[downlim], bin_edges[-1]])
        else:
            downlim = np.max(np.where(bin_edges <= faidown)[0])
            rangedown.append([bin_edges[downlim], bin_edges[nbd]])


    # 收集角度范围的bin index（可选，若后续用到）
    for rng in rangeup + rangedown:
        id1 = np.max(np.where(bin_edges <= rng[0])[0])
        id2 = np.max(np.where(bin_edges <= rng[1])[0]) - 1
        lim.extend(range(id1, id2 + 1))
    lim = np.unique(np.sort(lim))

    # 分类存储
    MEDIAN_a_candi, a_candi_fau, faia = [], [], []
    MEDIAN_u_candi, u_candi_fau, faiu = [], [], []

    for i in range(len(line_nums_1)):
        index = np.arange(line_nums_1[i, 0], line_nums_1[i, 1] + 1)
        vec = B[i]
        fai = np.degrees(np.arctan2(vec[0], vec[1]))
        if fai < 0:
            fai += 180
        
        # 给该断层赋随机颜色
        color = np.random.rand(3)
        candi_fau[index, 10:13] = color

        isInAnyRange = False
        for rng in rangeup:
            if rng[0] <= fai <= rng[1]:
                MEDIAN_a_candi.append(np.hstack((np.median(candi_fau[index, 0:3], axis=0),
                                          candi_fau[index[0], 13],  # 编号
                                          candi_fau[index, 0:3].shape[0])))  # 点数
                a_candi_fau.append(candi_fau[index])
                faia.append(fai)
                isInAnyRange = True
                break
        if not isInAnyRange:
            for rng in rangedown:
                if rng[0] <= fai <= rng[1]:
                    MEDIAN_a_candi.append(np.hstack((np.median(candi_fau[index, 0:3], axis=0),
                                              candi_fau[index[0], 13],
                                              candi_fau[index, 0:3].shape[0])))
                    a_candi_fau.append(candi_fau[index])
                    faia.append(fai)
                    isInAnyRange = True
                    break
        if not isInAnyRange:
            MEDIAN_u_candi.append(np.hstack((np.median(candi_fau[index, 0:3], axis=0),
                                      candi_fau[index[0], 13],
                                      candi_fau[index, 0:3].shape[0])))
            u_candi_fau.append(candi_fau[index])
            faiu.append(fai)

    # 转为 numpy 数组形式
    MEDIAN_u_candi = np.array(MEDIAN_u_candi)
    MEDIAN_a_candi = np.array(MEDIAN_a_candi)
    a_candi_fau = np.vstack(a_candi_fau) if a_candi_fau else np.empty((0, candi_fau.shape[1]))
    u_candi_fau = np.vstack(u_candi_fau) if u_candi_fau else np.empty((0, candi_fau.shape[1]))

    # 合并未归类断层 MEDIAN0 → MEDIAN1
    integra_fau = np.copy(candi_fau).astype(float)
    if MEDIAN_u_candi.size and MEDIAN_a_candi.size:
        for i in range(MEDIAN_u_candi.shape[0]):
            dists = np.linalg.norm(MEDIAN_a_candi[:, :3] - MEDIAN_u_candi[i, :3], axis=1)
            guiyi1 = np.max(dists)
            guiyi2 = np.max(MEDIAN_u_candi[i, 4] / MEDIAN_a_candi[:, 4])
            norm_dists = dists / guiyi1 + (MEDIAN_u_candi[i, 4] / MEDIAN_a_candi[:, 4]) / guiyi2
            idx_merge = np.argmin(norm_dists)

            M_a_label = MEDIAN_a_candi[idx_merge, 3]
            M_u_label = MEDIAN_u_candi[i, 3]

            rs = np.unique(integra_fau[integra_fau[:, 13] == M_a_label][:, 10:13], axis=0)
            integra_fau[integra_fau[:, 13] == M_u_label, 10:13] = rs[0]
            integra_fau[integra_fau[:, 13] == M_u_label, 13] = M_a_label

            MEDIAN_a_candi = np.delete(MEDIAN_a_candi, idx_merge, axis=0)
            if MEDIAN_a_candi.shape[0] == 0:
                break
    else:
        print("Warning: Some MEDIAN groups are empty, skipping merge.")

    # 排序并生成 line_nums_2（若需要）
    integra_fau = integra_fau[np.argsort(integra_fau[:, 13])]
    line_nums_2 = []
    for label in np.unique(integra_fau[:, 13]):
        indices = np.where(integra_fau[:, 13] == label)[0]
        line_nums_2.append([indices[0], indices[-1], label])
    line_nums_2 = np.array(line_nums_2)

    return u_candi_fau, a_candi_fau, line_nums_2, integra_fau

def Mtheo(candi_fault):

    def calc_bmemag(mag_sel, fBinning):   
        nLen = len(mag_sel)
        fMinMag = np.min(mag_sel)
        fMeanMag = np.mean(mag_sel)

        # 最大似然法计算 b 值
        fBValue = (1 / (fMeanMag - (fMinMag - (fBinning / 2)))) * np.log10(np.e)

        # b 值的标准差
        fStdDev = np.sum((mag_sel - fMeanMag) ** 2) / (nLen * (nLen - 1))
        fStdDev = 2.30 * np.sqrt(fStdDev) * fBValue**2

        # 计算 a 值
        fAValue = np.log10(nLen) + fBValue * fMinMag

        return fBValue, fStdDev, fAValue

    def calc_mc_max_curvature(mCatalog):
        if mCatalog.size == 0:
            return np.nan

        fMaxMagnitude = np.max(mCatalog)
        fMinMagnitude = np.min(mCatalog)

        vMagCenters = np.arange(fMinMagnitude, fMaxMagnitude + 0.001, 0.01)
        vMagEdges = np.round(np.arange(fMinMagnitude - 0.05, fMaxMagnitude + 0.05 + 0.01, 0.01), 2)
        # vMagEdges = np.arange(fMinMagnitude - 0.05, fMaxMagnitude + 0.05 + 0.01, 0.01,)
        vHist, _ = np.histogram(mCatalog, bins=vMagEdges)

        max_count = np.max(vHist)
        max_indices = np.where(vHist == max_count)[0]
        if len(max_indices) == 0:
            return np.nan

        # Option 1: First peak (recommended by seismological convention)
        fMc = round(vMagCenters[max_indices[0]],2)
        # Option 2: Average if you prefer to smooth multiple peaks
        # fMc = np.mean(vMagCenters[max_indices])
        
        return fMc
        

    ev_mag = candi_fault[:, 3]  # 4th column in MATLAB is index 3 in Python
    mc = calc_mc_max_curvature(ev_mag)
    mag_sel = ev_mag[ev_mag >= mc + 0.2]
    if len(mag_sel) == 0:
        return np.nan, np.nan, np.nan
    b, bstd, a = calc_bmemag(mag_sel, fBinning=0.01)
    N = 1
    M = (np.log10(N) - a) / b if b != 0 else np.nan
    return a, b, M

def Mag_DBSCAN(integra_fau, line_nums_2, Radius_model, colortemplate, minPoint, C):
    Fau_km = []
    MaxMw=[]
    n=1
    for i in range(1, int(np.max(integra_fau[:,13])) + 1):
        if i in line_nums_2[:, 2]:
            k = np.where(line_nums_2[:, 2] == i)[0][0]
            start_idx = int(line_nums_2[k, 0])
            end_idx = int(line_nums_2[k, 1]) + 1
            candi_fault = integra_fau[start_idx:end_idx, :].copy()
            while True:
                Mobs = np.max(candi_fault[:, 3])  # 最大震级
                if candi_fault.shape[0] > 100:
                    a, b, Mthre = Mtheo(candi_fault)
                    Mw = max(Mobs, Mthre)
                else:
                    Mw = Mobs


                radius = C * Radius_model(Mw)
                candi_fault = candi_fault[np.argsort(candi_fault[:, 3])[::-1]]  # 降序按震级排序
                coords = candi_fault[:, 0:3]

                db = DBSCAN(eps=radius, min_samples=minPoint).fit(coords)
                idx_5 = db.labels_
                if np.all(idx_5 == -1) or len(idx_5) == 0:
                    break
                MaxMw.append(Mw)
                
                # 聚类中标记为1类的事件提取
                subfau_km = candi_fault[idx_5 == 0, :].copy()
                subfau_km[:, 10] = colortemplate[i-1, 3*(n-1) + 0]
                subfau_km[:, 11] = colortemplate[i-1, 3*(n-1) + 1]
                subfau_km[:, 12] = colortemplate[i-1, 3*(n-1) + 2]
                subfau_km[:, 13] = n  # 聚类编号

                Fau_km.append(subfau_km)
                n += 1

                # 移除已聚类事件
                candi_fault = candi_fault[idx_5 != 0, :]
                if candi_fault.shape[0] == 0:
                    break

    MaxMw = np.array(MaxMw).reshape(-1, 1)
 
    Fau_km = np.vstack(Fau_km) if Fau_km else np.empty((0, integra_fau.shape[1]))
    if Fau_km.shape[0] == 0:
        print(f'C-value: {C} is too small, please increase it !')
        return None, None

    return Fau_km, MaxMw

def Radius_model(Mw):
    # 原始数据
    Mag = np.array([-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7])
    D_data = np.array([0.001, 0.0015, 0.002, 0.005, 0.01, 0.4, 0.45, 0.5, 0.5, 0.5, 0.5]) * 1e-2
    k_data = np.array([1.1, 1.11, 1.12, 1.15, 1.2, 3, 3.5, 5, 6.5, 8, 10])

    # --- 拟合 D (fault displacement)
    def D_model(mag, a, b, c, d):
        return a * (1.0 / (1 + np.exp(b * mag + c))) + d

    initial_guess_D = [1, 1, 1, 1]
    params_D, _ = curve_fit(D_model, Mag, D_data, p0=initial_guess_D, maxfev=10000)
    D = D_model(Mw, *params_D)

    # --- 拟合 k (L/W ratio)
    def k_model(mag, a, b, c, d):
        return d * np.log(1 + np.exp(a * mag + b)) + c

    initial_guess_k = [1, 1, 1, 1]
    params_k, _ = curve_fit(k_model, Mag, k_data, p0=initial_guess_k, maxfev=10000)
    k = k_model(Mw, *params_k)

    # --- 计算 radius
    radius = 0.001 * ((10**(1.5 * Mw + 9.1)) / (k**2 * 3e10 * D))**(1/3) / k
    return radius

def Geometrics_info(m1, subfau_km):
    # 置信水平
    conf = 0.9
    scale_el = np.sqrt(chi2.ppf(conf, df=3))
    
    # PCA 主轴分析
    pca = PCA(n_components=3)
    pca.fit(subfau_km[:, 0:3])
    coeff = pca.components_.T  # 每列是一个主轴方向
    latent = pca.explained_variance_
    
    # 长中短轴向量（上下端点）
    p1_u = scale_el * np.sqrt(latent[0]) * coeff[:, 0] + m1
    p2_u = scale_el * np.sqrt(latent[1]) * coeff[:, 1] + m1
    p3_u = scale_el * np.sqrt(latent[2]) * coeff[:, 2] + m1
    
    p1_d = m1 - scale_el * np.sqrt(latent[0]) * coeff[:, 0]
    p2_d = m1 - scale_el * np.sqrt(latent[1]) * coeff[:, 1]
    p3_d = m1 - scale_el * np.sqrt(latent[2]) * coeff[:, 2]
    
    # 向量
    V_a = p1_u - p1_d
    V_b = p2_u - p2_d
    V_c = p3_u - p3_d
    
    V_ah = np.array([V_a[0], V_a[1], 0])
    strike = np.arctan2(V_ah[0], V_ah[1]) * 180 / np.pi
    if strike > 180:
        strike -= 180
    elif strike < 0:
        strike += 180
    
    n = np.cross(V_a, V_c)
    dip = np.arctan2(np.linalg.norm(np.cross(n, [1, 0, 0])), np.dot(n, [1, 0, 0])) * 180 / np.pi
    if dip > 90:
        dip = 180 - dip
    
    # 投影向量
    V_ap = V_a - np.dot(V_a, n) / np.dot(n, n) * n
    V_bp = np.array([V_b[0], V_b[1], 0])
    
    # 椭球长短轴
    a = scale_el * np.sqrt(latent[0])
    b = scale_el * np.sqrt(latent[1])
    c = scale_el * np.sqrt(latent[2])
    
    Aspect3D = 1 - (b/a) * (c/a)
    
    # 生成椭球点云
    theta = np.linspace(0, 2 * np.pi, 20)
    phi = np.linspace(-np.pi/2, np.pi/2, 10)
    Theta, Phi = np.meshgrid(theta, phi)
    
    X = (m1[0] + 
         a * coeff[0, 0] * np.cos(Phi) * np.cos(Theta) +
         b * coeff[0, 1] * np.cos(Phi) * np.sin(Theta) +
         c * coeff[0, 2] * np.sin(Phi))
    
    Y = (m1[1] + 
         a * coeff[1, 0] * np.cos(Phi) * np.cos(Theta) +
         b * coeff[1, 1] * np.cos(Phi) * np.sin(Theta) +
         c * coeff[1, 2] * np.sin(Phi))
    
    Z = (m1[2] + 
         a * coeff[2, 0] * np.cos(Phi) * np.cos(Theta) +
         b * coeff[2, 1] * np.cos(Phi) * np.sin(Theta) +
         c * coeff[2, 2] * np.sin(Phi))
    
    X = X.flatten()
    Y = Y.flatten()
    Z = Z.flatten()
    
    ellipsoid_points = np.vstack((X, Y, Z)).T
    
    return (a, b, c,
            p1_u, p1_d,
            p2_u, p2_d,
            p3_u, p3_d,
            strike, dip,
            Aspect3D, ellipsoid_points)

def FaultModel(FaultParameters):
    '''
    '''

    Rectangle_points = []

    # 画断层矩形
    for i in range(len(FaultParameters)):
        A1 = FaultParameters[i, 6:9]
        A2 = FaultParameters[i, 9:12]
        B1 = FaultParameters[i, 12:15]
        B2 = FaultParameters[i, 15:18]

        L = A2 - A1
        S = B2 - B1

        S_unit = S / np.linalg.norm(S)
        S_half = 0.5 * np.linalg.norm(S)

        V1 = A1 - S_half * S_unit
        V2 = A1 + S_half * S_unit
        V3 = A2 + S_half * S_unit
        V4 = A2 - S_half * S_unit

        rectangle_points = np.vstack([V1, V2, V3, V4, V1])
        Rectangle_points.append(np.hstack([rectangle_points, np.full((rectangle_points.shape[0], 1), i)]))

    Rectangle_points = np.vstack(Rectangle_points)

    return Rectangle_points
