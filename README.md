# E2F_Python

Earthquake-to-Fault (E2F) pipeline.

---

## Windows/Linux (Create and activate environment)
conda create -n E2Fpy python=3.10  
conda activate E2Fpy  
conda install scipy scikit-learn pyvista matplotlib plotly numpy mkl mkl-service mkl-devel  

(Enter the project directory: xx\E2F_Python)

---

## Workflow

### 0. Hough Transform
Parameters
- -dx : Used in Hough transform algorithm. Default is 0, meaning each dimension of the space is divided into 64 parts
- -catalog : Your earthquake catalog path
- -hough : Path to hough-3d-lines-master  

python .\src\runE2F.py step0 -catalog .\example\catalog\ToC2ME.txt -hough .\hough-3d-lines-master\hough3dlines

### 1. Accepted and Unaccepted Fault Candidates
Parameters
- -m <MODE>
  - 1 -> One preferred fault orientation
  - 2 -> Two or more preferred fault orientations
- -pm <PBAD factor> (used in MODE 1) → Suggested value: 2
  (PBAD = Deviation of the preferred fault orientation)
- -mf <main fault orientation> (used in MODE 2) → Multiple fault orientations can be set
- -at <angular tolerance> (used in MODE 2) → Angular deviation of the main fault orientation (-mf)

1.1 One preferred fault orientation  
python .\src\runE2F.py step1 -m 1 -pm 2

1.2 Two or more preferred fault orientations  
python .\src\runE2F.py step1 -m 2 -mf 0 10 -at 10

### 2. Fault Segment Clustering
Parameters
- -mp <minpoint> : Minimum number of points in a cluster (default = 10)
- -c <scaling coefficient> : Adjust the clustering distance

python .\src\runE2F.py step2 -c 1 2 3 4 5 6 7 8 [Free number of c values, not limited to <= 8]

### 3. Fault Structure Modeling
Parameters
- -oc <optimal scaling coefficient> : An optimal c from step 2
- -savepath <output file saving path> : Default is .\output

python .\src\runE2F.py step3 -oc 3.5


## Fault Segment Clustering  
<img width="2118" height="1185" alt="FC8" src="https://github.com/user-attachments/assets/0cb9ae78-7b9d-4412-8e6b-87fb7fee77f8" />

## Fault Structure Modeling    
<img width="1355" height="1014" alt="FM3 5" src="https://github.com/user-attachments/assets/b1191f6e-7130-4c01-9345-f274144cb7d9" />
