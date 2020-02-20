#!/bin/bash
#PBS -q mamba
#PBS -l nodes=1:ppn=1:gpus=1
/users/adhere/HC/lab_1/p2_no-memcpy.out 100
/users/adhere/HC/lab_1/p2_no-memcpy.out 1000
/users/adhere/HC/lab_1/p2_no-memcpy.out 10000
/users/adhere/HC/lab_1/p2_memcpy.out 100
/users/adhere/HC/lab_1/p2_memcpy.out 1000
/users/adhere/HC/lab_1/p2_memcpy.out 10000

