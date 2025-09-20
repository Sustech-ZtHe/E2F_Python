# ----------------------------------------- #
#        E2F v0.0.1                         #
#        Main Function for E2F              #
#        Author: Zhengtao He                #
#        Email:                             #
#        Last update time: 6/29/2025        #
# ----------------------------------------- #
    # update log:
    # v0.0.1: Initial version, basic functions (SRL)
# ----------------------------------------- #
import os
import argparse
from pipelines import run_step0, run_step1, run_step2, run_step3

def main():
    '''

    '''

    # 定义通用参数的父解析器
    global_parser = argparse.ArgumentParser(add_help=False)
    global_parser.add_argument("-tempfile_path", type=str, default=os.path.join(os.getcwd(), "tempfile"))
    
    # 主解析器
    parser = argparse.ArgumentParser(description="E2F Pipeline Steps")
    subparsers = parser.add_subparsers(title="pipelines", dest="step")

    # step0
    parser_step0 = subparsers.add_parser("step0", parents=[global_parser], help="")
    parser_step0.add_argument("-catalog", required=True, type=str)
    parser_step0.add_argument("-hough", required=True, type=str)
    parser_step0.add_argument("-dx", type=float, help="Optional dx value for hough3dlines (Step 0)")
    parser_step0.set_defaults(func=run_step0)

    # step1
    parser_step1 = subparsers.add_parser("step1", parents=[global_parser], help="")
    parser_step1.add_argument("-m", required=True, type=int, choices=[1, 2], help="Fault classification mode (Step 1)")
    parser_step1.add_argument("-pm", type=float, help="PBAD_Multiple, used in mode 1 (Step 1)")
    parser_step1.add_argument("-mf", nargs="+", type=float, help="Main fault azimuth range, used in mode 2 (Step 1)")
    parser_step1.add_argument("-at", type=float, help="Angle tolerance, used in mode 2 (Step 1)")
    parser_step1.set_defaults(func=run_step1)

    # step2
    parser_step2 = subparsers.add_parser("step2", parents=[global_parser], help="")
    parser_step2.add_argument("-c", required=True, nargs="+", type=float, help="List of C values for clustering (Step 2)")
    parser_step2.add_argument("-mp", type=int, default=10, help="Minimum points for DBSCAN (Step 2)")
    parser_step2.add_argument("-color_num", type=int, default=10000)
    parser_step2.set_defaults(func=run_step2)

    # step3
    parser_step3 = subparsers.add_parser("step3", parents=[global_parser], help="")
    parser_step3.add_argument("-oc", required=True, type=float, help="Optimal C value")
    parser_step3.add_argument("-savepath", type=str, default=os.path.join(os.getcwd(), "output"), help="path for output files")
    parser_step3.set_defaults(func=run_step3)

    args = parser.parse_args()

    if args.step is None:
        parser.print_help()
    else:
        if not os.path.exists(args.tempfile_path):
            os.makedirs(args.tempfile_path)
        args.func(args)

if __name__ == "__main__":
    main()