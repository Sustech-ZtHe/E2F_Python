# %%
import os
import numpy as np
# %%
def read_catalog(filepath):
    """
    Read catalog file and convert WGS-84 to local Cartesian coordinates
    using minimum lat/lon as origin, purely with numpy (no geopy).
    """
    hypoloc = []
    with open(filepath, 'r') as file:
        for line in file:
            cols = line.strip().split()
            try:
                evid, year, month, day, hour, minute, second = map(float, cols[:7])
                lat, lon, depth, mag = map(float, cols[7:11])
                hypoloc.append([evid, year, month, day, hour, minute, second, lat, lon, depth, mag])
            except:
                continue

    hypoloc = np.array(hypoloc)

    # 取最小经纬度作为原点
    lat0 = np.min(hypoloc[:, 7])
    lon0 = np.min(hypoloc[:, 8])

    evla = hypoloc[:, 7]
    evlo = hypoloc[:, 8]
    evdp = hypoloc[:, 9]
    mag  = hypoloc[:, 10]

    # 地球半径近似值
    R = 6371.0  # km

    # x 方向（经度方向）: 考虑纬度缩放
    evlo_km = (evlo - lon0) * np.cos(np.deg2rad(lat0)) * 2 * np.pi * R / 360
    # y 方向（纬度方向）
    evla_km = (evla - lat0) * 2 * np.pi * R / 360
    # z 方向保持深度
    evdp_km = -evdp

    # 输出顺序同之前：x, y, z, mag, year, month, day, hour, minute, second
    hypoloc_km = np.column_stack((
        evlo_km,
        evla_km,
        evdp_km,
        mag,
        hypoloc[:, 1],  # year
        hypoloc[:, 2],  # month
        hypoloc[:, 3],  # day
        hypoloc[:, 4],  # hour
        hypoloc[:, 5],  # minute
        hypoloc[:, 6]   # second
    ))

    return hypoloc_km


def save_results(savepath, Fau_km, Optimal_FaultParameters):
    '''
    '''

    # --- 写 Fault_Segment_Clusterd.txt ---
    header1 = [
        '#1 Event X (km)',
        '#2 Event Y (km)',
        '#3 Depth',
        '#4 Magnitude',
        '#5-10 Year Month Day Hour Minute Second',
        '#11-13 R G B',
        '#14 Fault ID',
    ]
    file_path1 = os.path.join(savepath, 'Fault_Segment_Clusterd.txt')

    # 先写header
    with open(file_path1, 'w') as f:
        for line in header1:
            f.write(line + '\n')

    # 追加写数据
    with open(file_path1, 'a') as f:
        for row in Fau_km:
            line = ' '.join(f'{num:.6f}' for num in row)
            f.write(line + '\n')

    # --- 写 Fault_Segment_Modeling.txt ---
    header2 = [
        '#1 Fault ID',
        '#2-4 Centroid X (km) Centroid Y (km) Centroid Z (km)',
        '#5-7 Major Axis Upper Endpoint X (km) Major Axis Upper Endpoint Y (km) Major Axis Upper Endpoint Z (km)',
        '#8-10 Major Axis Lower Endpoint X (km) Major Axis Lower Endpoint Y (km) Major Axis Lower Endpoint Z (km)',
        '#11-13 Intermediate Axis Upper Endpoint X (km) Intermediate Axis Upper Endpoint Y (km) Intermediate Axis Upper Endpoint Z (km)',
        '#14-16 Intermediate Axis Lower Endpoint X (km) Intermediate Axis Lower Endpoint Y (km) Intermediate Axis Lower Endpoint Z (km)',
        '#17-19 Short Axis Upper Endpoint X (km) Short Axis Upper Endpoint Y (km) Short Axis Upper Endpoint Z (km)',
        '#20-22 Short Axis Lower Endpoint X (km) Short Axis Lower Endpoint Y (km) Short Axis Lower Endpoint Z (km)',
        '#23 Strike (deg)',
        '#24 Dip Angle (deg)',
        '#25 Optimal C Value',
        '#26 Number of Events constituting the fault',
    ]
    file_path2 = os.path.join(savepath, 'Fault_Segment_Modeling.txt')

    with open(file_path2, 'w') as f:
        for line in header2:
            f.write(line + '\n')

    Fault_Segment_Modeling = np.hstack([
        Optimal_FaultParameters[:, 29:30],   # 30th column (fault ID)
        Optimal_FaultParameters[:, 0:3],     # 1:3
        Optimal_FaultParameters[:, 6:24],    # 7:18
        Optimal_FaultParameters[:, 24:26],   # 25:26
        Optimal_FaultParameters[:, 27:29],   # 28:29
    ])

    with open(file_path2, 'a') as f:
        for row in Fault_Segment_Modeling:
            line = ' '.join(f'{num:.6f}' for num in row)
            f.write(line + '\n')
