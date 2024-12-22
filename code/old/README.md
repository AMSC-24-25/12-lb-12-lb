Steps of the code implementation
1) Sequential code
2) Implementation of class
3) Parallelization
4) Stuctured code (not parallelized)
5) Stuctured code parallelized


In order to run the code in this folder simply write: <br />
chmod +x compile_and_run.sh <br />
./compile_and_run.sh <version> [number_of_cores] <br />

and sostitute <version with the required version and [number of cores] with the number of cores. Since only 3 is parallelized you can write the number of cores only when you call LBM_3. You can also not insert the number of cores in the 3 case and it will take the maximum number of cores in your pc. <br />
Example of usage: <br />
./compile_and_run.sh 1 <br />
./compile_and_run.sh 2 <br />
./compile_and_run.sh 3 8 <br />
./compile_and_run.sh 3 5 <br />
./compile_and_run.sh 3 <br />
./compile_and_run.sh 4 <br />
