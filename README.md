# E2F_Python
Windows

   create and active environment

      conda create -n E2Fpy python=3.10 
      conda activate E2Fpy
      conda install scipy scikit-learn pyvista matplotlib plotly numpy mkl mkl-service mkl-devel 
      (Enter the road like: xx\E2F_Python)

   0. Hough transform 
      PARAMETERS:
      -dx <Used in Hough transform algorithm> default is 0, which means that each dimension of the space is divided into 64 parts
      -catalog <your earthquake catalog path>
      -hough <your hough-3d-lines-master path>
 
      python .\src\runE2F.py step0 -catalog .\example\catalog\ToC2ME.txt -hough .\hough-3d-lines-master\hough3dlines.exe

   1. "Accepted" and "Unaccepted" fault candidates
      PARAMETERS:
      -m <MODE> We set two modes for 
         a) one preferred fault orientation (-m 1)
         b) two or more preferred fault orientation (-m 2)
      -pm <PBAD factor, Used in MODE 1> We suggest set PBAD factor (-pm) as 2
         (PBAD: Deviation of the preferred fault orientation)
      -mf <some main fault orientation, Used in MODE 2> You can set multiple fault orientations
      -at <angular tolerance, Used in MODE 2> Angular deviation of the main fault orientation (-mf)

      1.2.1 For one preferred fault orientation
         python .\src\runE2F.py step1 -m 1 -pm 2
      1.2.2 For two or more preferred fault orientation
         python .\src\runE2F.py step1 -m 2 -mf 0 10 -at 10

   2. Fault segment clustering
      PARAMETERS:
      -mp <minpoint> The minimum number of points in a cluster and we set 10 as default
      -c <scaling coefficient> Adjust the clustering distance

      python .\src\runE2F.py step2 -c 1 2 3 4 5 6 7 8 [free number, not limited <= 8]
   
   3. Fault structure Modeling
      PARAMETERS: 
      -oc <optimal scaling coefficient> an optimal c from step 2
      -savepath <outputfile saving path> default is .\output

      python .\src\runE2F.py step3 -oc 3.5 [one number]



Linux

   create and active environment

      conda create -n E2Fpy python=3.10 
      conda activate E2Fpy
      conda install scipy scikit-learn pyvista matplotlib plotly numpy mkl mkl-service mkl-devel 
      (Enter the road like: xx\E2F_Python)

   0. Hough transform 
      PARAMETERS:
      -dx <Used in Hough transform algorithm> default is 0, which means that each dimension of the space is divided into 64 parts
      -catalog <your earthquake catalog path>
      -hough <your hough-3d-lines-master path>
 
      python ./src/runE2F.py step0 -catalog ./example/catalog/ToC2ME.txt -hough ./hough-3d-lines-master/hough3dlines

   1. "Accepted" and "Unaccepted" fault candidates
      PARAMETERS:
      -m <MODE> We set two modes for 
         a) one preferred fault orientation (-m 1)
         b) two or more preferred fault orientation (-m 2)
      -pm <PBAD factor, Used in MODE 1> We suggest set PBAD factor (-pm) as 2
         (PBAD: Deviation of the preferred fault orientation)
      -mf <some main fault orientation, Used in MODE 2> You can set multiple fault orientations
      -at <angular tolerance, Used in MODE 2> Angular deviation of the main fault orientation (-mf)

      1.2.1 For one preferred fault orientation
         python ./src/runE2F.py step1 -m 1 -pm 2
      1.2.2 For two or more preferred fault orientation
         python ./src/runE2F.py step1 -m 2 -mf 0 10 -at 10

   2. Fault segment clustering
      PARAMETERS:
      -mp <minpoint> The minimum number of points in a cluster and we set 10 as default
      -c <scaling coefficient> Adjust the clustering distance

      python ./src/runE2F.py step2 -c 1 2 3 4 5 6 7 8 [free number, not limited <= 8]
   
   3. Fault structure Modeling
      PARAMETERS: 
      -oc <optimal scaling coefficient> an optimal c from step 2
      -savepath <outputfile saving path> default is .\output

      python ./src/runE2F.py step3 -oc 3.5 [one number]

