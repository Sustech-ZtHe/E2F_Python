# %%
import os
import shutil
import platform
import numpy as np
from base import faults
from dataio import read_catalog, save_results
from utils import Candidate_fault_classify, Mag_DBSCAN, Radius_model, FaultModel
from graphics import plot_linear_identification, plot_fault_clutering, plot_ellpisoid_fitting,  \
    plot_fault_dip, plot_fault_azi, plot_aspect_ratio, plot_rms, plot_metrics_evaluation, plot_fault_rectangle
# %%
# -- step0 array clarification -- #
# catalog N*11
# evid, year, month, day, hour, minute, second, lat, lon, depth, mag
# hypo_km N*10
# x_km, y_km, z_km, mag, year, month, day, hour, minute, second
def run_step0(args):
    '''
    Run Step 0: Hough Transform to identify candidate lines from seismic catalog.
    '''

    print("=== Running Step 0 (Hough Transform) ===")

    hypo_km = read_catalog(args.catalog)
    print(f"Read {len(hypo_km)} events from catalog: {args.catalog}")
    
    # 先清空 args.tempfile_path 下的文件
    if os.path.exists(args.tempfile_path):
        for f in os.listdir(args.tempfile_path):
            file_path = os.path.join(args.tempfile_path, f)
            if os.path.isfile(file_path):
                os.remove(file_path)

    system = platform.system()
    if system not in ["Linux", "Windows"]:
        raise ValueError("Unsupported operating system. Only Linux and Windows are supported.")
    
    if system == "Windows":
        catalog_path = args.catalog.replace('\\', '/')
    elif system == "Linux":
        catalog_path = args.catalog


    output_file = os.path.join(args.tempfile_path, catalog_path.split('/')[-1] + "_km.txt")

    np.savetxt(output_file, hypo_km, fmt="%.6f", delimiter="\t")

    # ------------------- hough3dlines ---------------------------- #
    if not os.path.exists(output_file):
        raise FileNotFoundError(f"{output_file} does not exist. Please check it.")
    
    if args.dx == None:
        command = f"{args.hough} {output_file}"
    else:
        if args.dx < 0: 
            raise ValueError("dx must be >= 0.")
        command = f"{args.hough} -dx {str(args.dx)} {output_file}"
    
    if system == "Linux" or system == "Windows":
        os.system(f"{command}")
    else:
        raise ValueError("Hough tranform command only tested on Linux and Windows system!")

     # ------------------------------------------------------------ #

    print(f"Save output files '{catalog_path.split('/')[-1] + '_km.txt'}' 'ab.output.txt' 'sp.output.txt' to {args.tempfile_path}")
    print("=== E2F-step0: Finished ===")

# -- step1 array clarification -- #
# u_candi_fau: unaccepted lineament N1*14
# a_candi_fau: accepted lineament N2*14
# integra_fau: (N1+N2)*14
# line_nums_2: ?*3
def run_step1(args):
    '''
    '''
    
    print("=== Running Step 1 (Lineament Identification and Classificaito) ===")

    # --- 加载H3D识别的候选断层(candi_fau)和其3D直线方程参数(line_ab) --- #
    file_path_ab = os.path.join(args.tempfile_path, "ab.output.txt")
    file_path_sp = os.path.join(args.tempfile_path, "sp.output.txt")

    if (not os.path.exists(file_path_ab)) or (not os.path.exists(file_path_sp)):
        raise FileNotFoundError("Required files for Step 1 not found. Please run Step first.")
    else:
        line_ab = np.loadtxt(file_path_ab)
        candi_fau = np.loadtxt(file_path_sp)

    # ---------------------------------------------------------------- #
    if args.m == 1:
        if args.pm is None:
            raise ValueError("PBAD_Multiple (-pm) must be provided for mode 1.")
        if args.mf is not None:
            print("Warning: Main fault azimuth range (-mf) will be ignored in mode 1.")
        if args.at is not None:
            print("Warning: Angle tolerance (-at) will be ignored in mode 1.")
        PBAD_Multiple = args.pm
        main_fault = None
        angle_tolerance = None
    elif args.m == 2:
        if (args.mf is None) or (args.at is None):
            raise ValueError("Main fault azimuth range (-mf) and angle tolerance (-at) must be provided for mode 2.")
        if args.pm is not None:
            print("Warning: PBAD_Multiple (-pm) will be ignored in mode 2.")
        PBAD_Multiple = None
        main_fault = args.mf
        angle_tolerance = args.at
    else:
        raise ValueError("unrecognized Mode.")
    
    u_candi_fau, a_candi_fau, line_nums_2, integra_fau = Candidate_fault_classify(PBAD_Multiple, line_ab, candi_fau, args.m, main_fault, angle_tolerance)
    # ---------------------------------------------------------------- #
    
    candi_linear = {
        'a_candi_fau': a_candi_fau,
        'u_candi_fau': u_candi_fau,
        'integra_fau': integra_fau,
        'line_nums_2': line_nums_2}
    np.save(os.path.join(args.tempfile_path, 'candi_linear.npy'), candi_linear, allow_pickle=True)
    print(f"Save temp dict 'candi_linear' as {os.path.join(args.tempfile_path, 'candi_linear.npy')}")
    
    plot_linear_identification(u_candi_fau, a_candi_fau)

    print("E2F-step1: Finished.")

# -- step2 array clarification -- #
# class fault
# Fau_km ?*14
# MaxMw  ?*1
# FaultParameters ?*30
# median_xyz, a, b, c, p1_u, p1_d, p2_u, p2_d, p3_u, p3_d, az, dip, aspect3d, self.cvalue, sp_len, sp_col14
# Aspect3D ?*2 
# sp_len, aspect3d
# Magrecord ?*2
# mag n
# ell ?*4
# a b c n?
def run_step2(args):
    '''
    '''

    print("=== Running Step 2 (Fault Segment Clustering) ===")

    try:
        candi_linear = np.load(os.path.join(args.tempfile_path, 'candi_linear.npy'), allow_pickle=True).item()
    except Exception as e:
        raise FileNotFoundError("Required files for Step 2 not found. Please run previous Steps first.") from e
    
    if args.mp <= 4:
        print("Minimum points for DBSCAN (-mp) must be > 4. Reset to default value 10.")
        minpoints = 10
    else:
        minpoints = args.mp
    np.save(os.path.join(args.tempfile_path, 'minPoint.npy'), args.mp)
    print(f"Save output files 'minPoint.npy' to {args.tempfile_path}")

    if not os.path.exists(os.path.join(args.tempfile_path, 'colortemp.npy')):
        colortemp = np.random.rand(args.color_num, args.color_num)
        np.save(os.path.join(args.tempfile_path, 'colortemp.npy'), colortemp)
        print(f"Create new color array ({args.color_num}*{args.color_num}) and saved to {os.path.join(args.tempfile_path, 'colortemp.npy')}")
    else:
        colortemp = np.load(os.path.join(args.tempfile_path, 'colortemp.npy'))
        print(f"Using existing color array ({colortemp.shape[0]}*{colortemp.shape[1]}) {os.path.join(args.tempfile_path, 'colortemp.npy')}")
        
    # ---------------------------------------------------------------- #
    
    cvalues = np.array(args.c)
    faults_objects = []
    print("Processing ......")
    for idx, value in enumerate(cvalues):
        print(f"{idx+1}/{len(cvalues)}: c={value}")
        Fau_km, MaxMw = Mag_DBSCAN(candi_linear['integra_fau'], candi_linear['line_nums_2'], Radius_model, colortemp, minpoints, value)
        if Fau_km is None or Fau_km.shape[0] == 0:
            print(f'C={value} No clustered events found.')
        else:
            faults_objects.append(faults(value, Fau_km, MaxMw))
    # ---------------------------------------------------------------- #

    total_events = candi_linear['integra_fau'].shape[0]
    plot_fault_clutering(faults_objects, total_events)
    plot_ellpisoid_fitting(faults_objects, total_events)
    plot_fault_dip(faults_objects)
    plot_fault_azi(faults_objects)
    plot_aspect_ratio(faults_objects)
    plot_rms(faults_objects)
    plot_metrics_evaluation(faults_objects, total_events)
    # ---------------------------------------------------------------- #

    print("E2F-step2: Finished.")

# -- step3 array clarification -- #
# Rectangle_points
def run_step3(args):
    '''
    '''

    print("=== Running Step 3 (Fault Model Visualization) ===")
    try:
        candi_linear = np.load(os.path.join(args.tempfile_path, 'candi_linear.npy'), allow_pickle=True).item()
        colortemp = np.load(os.path.join(args.tempfile_path, 'colortemp.npy'))
        minpoints = np.load(os.path.join(args.tempfile_path, 'minPoint.npy')).item()
    except Exception as e:
        raise FileNotFoundError("Required files for Step 3 not found. Please run previous Steps first.") from e

    Fau_km, MaxMw = Mag_DBSCAN(candi_linear['integra_fau'], candi_linear['line_nums_2'], Radius_model, colortemp, minpoints, args.oc)
    if Fau_km is None or Fau_km.shape[0] == 0:
        print(f'C={args.oc} No clustered events found. Choose optimal values for C.')
    else:
        optimal_faults = faults(args.oc, Fau_km, MaxMw)
    Rectangle_points = FaultModel(optimal_faults.FaultParameters)
    if not os.path.exists(args.savepath):
        os.makedirs(args.savepath)
    save_results(args.savepath, optimal_faults.Fau_km, optimal_faults.FaultParameters)
    print(f"Saved files 'Fault_Segment_Clusterd.txt' 'Fault_Segment_Modeling.txt' to {args.savepath}")

    plot_fault_rectangle(Rectangle_points, optimal_faults.Fau_km, candi_linear['integra_fau'].shape[0])
    
    print("E2F-step3: Finished.")
# %%