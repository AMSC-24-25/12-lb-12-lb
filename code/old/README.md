Steps of the code implementation
1) Sequential code
2) Implementation of class
3) Parallelization
4) Optimized code


In order to run the code in this folder simply write \\
chmod +x compile_and_run.sh\\
./compile_and_run.sh <version> [number_of_cores]\\

and sostitute <version with the required version and [number of cores] with the number of cores. Since only 3 is parallelized you can write the number of cores only when you call LBM_3. You can also not insert the number of cores in the 3 case and it will take the maximum number of cores in your pc\\
Example of usage\\
./compile_and_run.sh 1\\
./compile_and_run.sh 2\\
./compile_and_run.sh 3 8\\
./compile_and_run.sh 3 5\\
./compile_and_run.sh 3\\
