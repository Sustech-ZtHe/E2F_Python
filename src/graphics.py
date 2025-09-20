import numpy as np 
import pyvista as pv
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull


# --------------------STEP1--------------------- #
def plot_linear_identification(u_candi_fau, a_candi_fau):
    
    pointsU = u_candi_fau[:, 0:3] if u_candi_fau.size > 0 else np.empty((0, 3))
    colorsU = u_candi_fau[:, 10:13] if u_candi_fau.size > 0 else np.empty((0, 3))

    pointsA = a_candi_fau[:, 0:3] if a_candi_fau.size > 0 else np.empty((0, 3))
    colorsA = a_candi_fau[:, 10:13] if a_candi_fau.size > 0 else np.empty((0, 3))

    cloudU = pv.PolyData(pointsU)
    if pointsU.shape[0] > 0:
        cloudU['colors'] = colorsU

    cloudA = pv.PolyData(pointsA)
    if pointsA.shape[0] > 0:
        cloudA['colors'] = colorsA

    plotter = pv.Plotter(shape=(1, 2), window_size=(1600, 900))

    plotter.subplot(0, 1)
    plotter.add_text("Unaccepted Faults", font_size=12, position='upper_edge')
    if pointsU.shape[0] > 0:
        plotter.add_points(cloudU, scalars='colors', rgb=True, 
                        point_size=5.0, render_points_as_spheres=True)
    plotter.set_background("white")

    plotter.show_bounds(
        grid='back',
        location='origin',
        all_edges=True,
        color='black',
        font_size=12,
        ticks='outside',
    )
    plotter.view_isometric()   # 斜视角，能看到 XYZ 三个轴

    plotter.subplot(0, 0)
    plotter.add_text("Accepted Faults", font_size=12, position='upper_edge')
    if pointsA.shape[0] > 0:
        plotter.add_points(cloudA, scalars='colors', rgb=True, 
                        point_size=5.0, render_points_as_spheres=True)
    plotter.set_background("white")

    plotter.show_bounds(
        grid='back',
        location='origin',
        all_edges=True,
        color='black',
        font_size=12,
        ticks='outside',
    )
    plotter.view_isometric()   # 斜视角，能看到 XYZ 三个轴

    plotter.show(title="Fault Classification Comparison")

# --------------------STEP2--------------------- #
def plot_fault_clutering(faults, total_events, max_cols=4):
    '''
    '''

    number_cvalue = len(faults)
    if number_cvalue == 0:
        print("No results to plot.")
        return
    else:
        cols = int(min(number_cvalue, max_cols))
        rows = int(np.ceil(number_cvalue / cols))
        plotter = pv.Plotter(shape=(rows, cols), window_size=(1600, 900))

    for idx, fault in enumerate(faults):

        if fault.Fau_km is None or fault.Fau_km.shape[0] == 0:
            used_events = 0
            fault_counts= 0
        else:
            used_events = fault.Fau_km.shape[0]
            fault_counts= int(fault.Fau_km[-1,-1])

        unused_events = total_events - used_events

        points = fault.Fau_km[:, 0:3] if used_events > 0 else np.empty((0, 3))
        colors = fault.Fau_km[:, 10:13] if used_events > 0 else np.empty((0, 3))

        cloud = pv.PolyData(points)
        cloud['colors'] = colors

        i_row = idx // max_cols
        i_col = idx % max_cols

        plotter.subplot(i_row, i_col)
        plotter.add_text(f"C = {fault.cvalue}", font_size=12, position='upper_edge')
        plotter.add_points(cloud, scalars='colors', rgb=True, point_size=5.0, render_points_as_spheres=True)
        plotter.set_background("white")
        plotter.show_bounds(
            grid='back',
            location='origin',
            all_edges=True,
            color='black',
            font_size=12,
            ticks='outside',
        )
        plotter.view_isometric()

        info_text = f"Used: {used_events}\nCandidate faults: {fault_counts}\nRemaining: {unused_events}"
        plotter.add_text(info_text, position='lower_right', font_size=12)

    plotter.show(title='Fault Segment Clustering')

def _ellipsoid_surf_(ell, plotter):
    unique_ids = np.unique(ell[:, -1])
    for i in unique_ids:
        num = np.where(ell[:, -1] == i)[0]
        points = ell[num, 0:3]

        if points.shape[0] < 4:
            continue  # 至少 4 点才能 ConvexHull

        hull = ConvexHull(points)

        faces = []
        for simplex in hull.simplices:
            faces.append(3)
            faces.extend(simplex.tolist())

        faces = np.array(faces)

        mesh = pv.PolyData(points, faces)

        plotter.add_mesh(mesh, color='red', opacity=0.3, show_edges=False)

def plot_ellpisoid_fitting(faults, total_events, max_cols=4):
    '''
    '''

    number_cvalue = len(faults)
    if number_cvalue == 0:
        print("No results to plot.")
        return
    else:
        cols = int(min(number_cvalue, max_cols))
        rows = int(np.ceil(number_cvalue / cols))
        plotter = pv.Plotter(shape=(rows, cols), window_size=(1600, 900))

    for idx, fault in enumerate(faults):
        
        if fault.Fau_km is None or fault.Fau_km.shape[0] == 0:
            used_events = 0
            fault_counts= 0
        else:
            used_events = fault.Fau_km.shape[0]
            fault_counts= int(fault.Fau_km[-1,-1])

        unused_events = total_events - used_events

        points = fault.Fau_km[:, 0:3] if used_events > 0 else np.empty((0, 3))
        colors = fault.Fau_km[:, 10:13] if used_events > 0 else np.empty((0, 3))

        cloud = pv.PolyData(points)
        cloud['colors'] = colors

        i_row = idx // max_cols
        i_col = idx % max_cols

        plotter.subplot(i_row, i_col)
        plotter.add_points(cloud, scalars='colors', rgb=True, point_size=5.0, render_points_as_spheres=True)
        info_text = f"Used: {used_events}\nCandidate faults: {fault_counts}\nRemaining: {unused_events}"
        plotter.add_text(info_text, position='lower_right', font_size=12)

        _ellipsoid_surf_(fault.ell, plotter)

        plotter.set_background("white")
        plotter.add_text(f"C = {fault.cvalue}", font_size=12, position='upper_edge')

        plotter.show_bounds(
            grid='back',
            location='origin',
            all_edges=True,
            color='black',
            font_size=12,
            ticks='outside',
        )
    plotter.view_isometric()  

    plotter.show(title='Fault Fitting')

def plot_fault_dip(faults, max_cols=4):
    '''
    '''

    number_cvalue = len(faults)
    if number_cvalue == 0:
        print("No results to plot.")
        return
    else:
        cols = int(min(number_cvalue, max_cols))
        rows = int(np.ceil(number_cvalue / cols))
        fig, axes = plt.subplots(rows, cols, subplot_kw={'projection': 'polar'}, figsize=(16, 8))
    axes = axes.flatten() if number_cvalue > 1 else [axes]
    
    maxdip = []
    for fault in faults:
        dipdeg = fault.FaultParameters[:, 25]
        hist_counts, _ = np.histogram(np.radians(dipdeg), bins=np.radians(np.arange(0, 100, 10)))
        maxdip.append(np.max(hist_counts))
    maxrlim = max(maxdip)

    for idx, fault in enumerate(faults):

        DipA = fault.FaultParameters[:, 25]
        _DipStat_ax_(axes[idx], DipA, maxrlim)
        axes[idx].set_title(f'C: {fault.cvalue}', fontsize=15)

        axes[idx].set_theta_zero_location("E")  # 0° 在正东
        axes[idx].set_theta_direction(1)        # 逆时针为正
        axes[idx].tick_params(axis='x', labelsize=20)  # 极坐标角度刻度
        axes[idx].tick_params(axis='y', labelsize=20)  # 半径刻度

    for i in range(idx + 1, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

def plot_fault_azi(faults, max_cols=4):
    '''
    '''

    number_cvalue = len(faults)
    if number_cvalue == 0:
        print("No results to plot.")
        return
    else:
        cols = int(min(number_cvalue, max_cols))
        rows = int(np.ceil(number_cvalue / cols))
        fig, axes = plt.subplots(rows, cols, subplot_kw={'projection': 'polar'}, figsize=(16, 8))
    axes = axes.flatten() if number_cvalue > 1 else [axes]

    maxaz = []
    for fault in faults:
        azdeg = fault.FaultParameters[:, 24]  # 28列为索引27, 25列为索引24
        counts, _ = np.histogram(np.radians(azdeg), bins=18)
        maxaz.append(counts.max())
    max_maxaz = max(maxaz)

    for idx, fault in enumerate(faults):

        Azimuth = fault.FaultParameters[:, 24]

        combined_azimuth = np.concatenate([Azimuth, Azimuth - 180])
        combined_azimuth = combined_azimuth % 360  # 保证在0-360范围内

        _AzimuthStat_ax_(axes[idx], combined_azimuth, max_maxaz)
        axes[idx].set_title(f'C: {fault.cvalue}', fontsize=15)
        axes[idx].tick_params(axis='x', labelsize=20)  # 极坐标角度刻度
        axes[idx].tick_params(axis='y', labelsize=20)  # 半径刻度

    for i in range(idx + 1, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

def plot_aspect_ratio(faults, max_cols=4):
    '''
    '''

    number_cvalue = len(faults)
    if number_cvalue == 0:
        print("No results to plot.")
        return
    else:
        cols = int(min(number_cvalue, max_cols))
        rows = int(np.ceil(number_cvalue / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(16, 8))
    axes = axes.flatten() if number_cvalue > 1 else [axes]

    maxylim = []
    bin_edges = np.arange(0, 1.1, 0.1)
    for fault in faults:
        counts, _ = np.histogram(fault.Aspect3D[:, 1], bins=bin_edges)
        maxylim.append(counts.max())
    maxylim_value = max(maxylim)

    for idx, fault in enumerate(faults):
        axes[idx].hist(fault.Aspect3D[:, 1], bins=bin_edges,
                    linewidth=2, color=[203/255, 238/255, 249/255],
                    edgecolor=[66/255, 146/255, 197/255])

        axes[idx].set_ylim([0, maxylim_value + 5])
        axes[idx].set_xlabel('Aspect Ratio', fontsize=20)
        axes[idx].set_ylabel('Count', fontsize=20)
        axes[idx].tick_params(axis='both', labelsize=20, width=1.5)
        axes[idx].spines['top'].set_linewidth(1.5)
        axes[idx].spines['right'].set_linewidth(1.5)
        axes[idx].spines['bottom'].set_linewidth(1.5)
        axes[idx].spines['left'].set_linewidth(1.5)
        axes[idx].set_title(f"C = {fault.cvalue}", fontsize=15)

    for i in range(idx + 1, len(axes)):
        axes[i].axis('off')
        
    plt.tight_layout()
    plt.show()

def plot_rms(faults, max_cols=4, mi=4):
    '''
    '''

    number_cvalue = len(faults)
    if number_cvalue == 0:
        print("No results to plot.")
        return
    else:
        cols = int(min(number_cvalue, max_cols))
        rows = int(np.ceil(number_cvalue / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(16, 8))
    axes = axes.flatten() if number_cvalue > 1 else [axes]    

    for idx, fault in enumerate(faults):
        d1 = np.min(fault.Motheo)
        d2 = np.max(fault.Motheo)
        dx = (d2 - d1) / 20
        d_range = np.arange(d1, d2 + dx, dx)

        # RSM = 1 线
        axes[idx].loglog(fault.Motheo, fault.Motheo, 'k-', linewidth=2, label='RSM=1')

        # RSM=1-mi/10
        axes[idx].loglog(d_range, (10 ** (1 - mi / 10)) * d_range ** (1 - mi / 10),
                    'k:', linewidth=1.5, label=f'RSM={1 - mi / 10:.1f}')

        # RSM=1+mi/10
        axes[idx].loglog(d_range, (10 ** (1 + mi / 10)) * d_range ** (1 + mi / 10),
                    'k--', linewidth=1.5, label=f'RSM={1 + mi / 10:.1f}')
        
        # Cumulative Mo of Events
        axes[idx].loglog(fault.Motheo, fault.Moreal, '+', markersize=8, linewidth=1.5,
                    markeredgecolor=np.array([161, 217, 156]) / 255)
        
        axes[idx].set_xlabel('Ellipsoid-based Mo', fontsize=20)
        axes[idx].set_ylabel('Cumulative Mo of Events', fontsize=20)
        axes[idx].tick_params(axis='both', labelsize=20, width=1)
        axes[idx].spines['top'].set_linewidth(1)
        axes[idx].spines['right'].set_linewidth(1)
        axes[idx].spines['bottom'].set_linewidth(1)
        axes[idx].spines['left'].set_linewidth(1)
        axes[idx].legend(loc='lower right', fontsize=10, frameon=False)
        axes[idx].set_title(f"C= {fault.cvalue}", fontsize=15)
        
    for i in range(idx + 1, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def plot_metrics_evaluation(faults, total_events, up=0.6):
    '''
    '''

    sumevents = []
    YUPmax = []
    for idx, fault in enumerate(faults):
        subflattening = fault.Aspect3D[:, 1]
        sumevents.append(np.sum(fault.Aspect3D[:, 0]))

        countse, edgese = np.histogram(subflattening, bins=[0, 0.8, 1])
        counter0, edgeer0 = np.histogram(fault.rsm, bins=[0, up, 1, 1 - up + 1])

        countse = countse[::-1]
        counter0 = counter0[::-1]

        yup = max(counter0[0] + counter0[1], countse[1])
        YUPmax.append(yup * 1.5)

    xtick = []
    xticklab = []
    fig, axes = plt.subplots(1, 1, figsize=(16, 8))
    for idx, fault in enumerate(faults):
        subflattening = fault.Aspect3D[:, 1]
        sumevents.append(np.sum(fault.Aspect3D[:, 0]))

        countse, edgese = np.histogram(subflattening, bins=[0, 0.8, 1])
        counter0, edgeer0 = np.histogram(fault.rsm, bins=[0, up, 1, 1 - up + 1])
        countse = countse[::-1]
        counter0 = counter0[::-1]
        
        edge = 0 + idx * 0.5
        xtick.append(edge)
        axes.bar(edge, countse[0], width=0.1, color=np.array([203, 238, 249]) / 255,
                edgecolor=np.array([66, 146, 197]) / 255, linewidth=3)
        axes.bar(edge + 0.1, counter0[0] + counter0[1], width=0.1, color=np.array([161, 217, 156]) / 255,
                edgecolor=np.array([5, 165, 158]) / 255, linewidth=3)
        axes.bar(edge + 0.2, np.sum(fault.Aspect3D[:, 0]) / total_events * max(YUPmax), width=0.1,
                color=np.array([143, 38, 126]) / 255, edgecolor=np.array([177, 49, 51]) / 255, linewidth=3)

        xticklab.append(f'C={fault.cvalue}')

    axes.set_ylim([0, max(YUPmax)])
    axes.legend(['3-D Aspect Ratio(0.8~1)', 'RSM (0.6~1.4)', 'Utilization rate of events'], frameon=False)
    axes.set_xticks(xtick, xticklab, fontsize=20)
    axes.set_ylabel('High RSM/3-D Aspect Ratio Counts', fontsize=20)
    axes.tick_params(axis='y', which='major', direction='in', labelsize=20)

    axes.grid(False)

    ax2 = axes.twinx()
    ax2.set_ylim(0, 100)
    ax2.tick_params(axis='y', labelsize=20, colors=np.array([143, 38, 126]) / 255)
    ax2.set_ylabel('Utilization rate (%)', fontsize=20, color=np.array([143, 38, 126]) / 255)

    plt.show()
    
def _DipStat_ax_(ax, DipAng, maxrlim=None):

    DipA_rad = np.radians(DipAng)
    bin_edges = np.radians(np.arange(0, 100, 10))
    counts, _ = np.histogram(DipA_rad, bins=bin_edges)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    ax.bar(bin_centers, counts, width=np.radians(10),
           color=[70/255, 130/255, 180/255], edgecolor='k', linewidth=1.5)

    ax.set_theta_zero_location("N")    # 90度朝上
    ax.set_theta_direction(-1)         # 顺时针
    ax.set_thetalim(0, np.pi / 2)      # 只显示 0-90 度
    ax.grid(linestyle='--', linewidth=2, color='black')
    ax.set_xticks(np.radians(np.arange(0, 100, 10)))
    ax.set_xticklabels([f"{deg}" for deg in np.arange(0, 100, 10)])
    ax.set_yticks(np.linspace(0, maxrlim, 5).astype(int))

    if maxrlim is not None:
        ax.set_rmax(maxrlim)
    else:
        uprlim = np.max(counts)
        rticks = np.arange(0, uprlim + 1, max(1, uprlim // 5))
        ax.set_yticks(rticks)
        ax.set_rmax(uprlim)

def _AzimuthStat_ax_(ax, AzimuthAng, maxrlim=None):
    Azimuth_rad = np.radians(AzimuthAng)
    bin_edges = np.linspace(0, 2 * np.pi, 37)
    counts, _ = np.histogram(Azimuth_rad, bins=bin_edges)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    ax.bar(bin_centers, counts, width=(2 * np.pi / 36),
           color=[220/255, 20/255, 60/255], edgecolor='k', linewidth=1.5)

    ax.set_theta_zero_location("N")  # 0度朝上
    ax.set_theta_direction(-1)       # 顺时针方向
    ax.grid(linestyle='--', linewidth=2, color='black')
    ax.tick_params(labelsize=12)

    ax.set_xticks(np.radians(np.arange(0, 360, 30)))
    ax.set_xticklabels([f"{deg}°" for deg in np.arange(0, 360, 30)])

    if maxrlim is not None:
        ax.set_rmax(maxrlim)
    else:
        uprlim = np.max(counts)
        rticks = np.arange(0, uprlim + 1, max(1, uprlim // 5))
        ax.set_yticks(rticks)
        ax.set_rmax(uprlim)


# --------------------STEP3--------------------- #
def plot_fault_rectangle(rectangle, Fau_km, total_events):
    '''
    '''

    plotter = pv.Plotter()
    for item in np.unique(rectangle[:, 3]):
        rectangle_points = rectangle[rectangle[:, 3]==item, 0:3]
        line = pv.lines_from_points(rectangle_points)
        plotter.add_mesh(line, color='black', line_width=2)

    points = Fau_km[:, 0:3]
    colors = Fau_km[:, 10:13]  # RGB in [0~1]
    if np.max(colors) > 1.0:
        colors = colors / 255.0  # 如果是 0~255，归一化

    point_cloud = pv.PolyData(points)
    point_cloud['colors'] = colors

    plotter.add_mesh(point_cloud, scalars='colors', rgb=True, point_size=6, render_points_as_spheres=True)

    plotter.view_isometric() 
    plotter.set_background(color='white')
    plotter.show_grid()
    plotter.show_axes()
    plotter.enable_anti_aliasing()

    clustered_events = Fau_km.shape[0]
    remaining_events = total_events - clustered_events
    plotter.add_text(
        f'Fault candidates = {int(Fau_km[-1, -1])}\n'
        f'Clustered events = {clustered_events}\n'
        f'Remaining events = {remaining_events}',
        font_size=12, position='lower_right'
    )

    plotter.show('Fault Structure Modeling')