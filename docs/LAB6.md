---
layout: default
title: VI.Explore ACCEL-SIM and Cache
nav_order: 8
---

# VI Explore ACCEL-SIM and Cache
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}
---

{: .outline}
> At this part, you will learn ACCEL-SIM and knowledge about cache

![ACCEL_SIM](../assets/images/accel-sim.svg)

[ACCEL-SIM](https://accel-sim.github.io/) is a new generation of GPGPU-SIM. The main difference between them is that ACCEL-SIM proposes a new energy model and SASS front end. And ACCEL-SIM provides many python scripts to simplify the execution and output data collection.

## [Build ACCEL-SIM](https://github.com/accel-sim/accel-sim-framework)

Recommend using Docker

{: .highlight}
> You may wonder why we(I) recommend using Docker. 
> Imagine that you need to install many apps and each app rely on different envs (for example, A => gcc8 and B => gcc10). 
> You will soon find that it's very complicated/impossible to build an env that all apps are compatible with each other.
> Then you will think of Docker in which each env is independent from each other.

To get docker image of ACCEL-SIM env
```
docker pull accelsim/ubuntu-18.04_cuda-11
```

To get ACCEL-SIM
```
git clone -b dev https://github.com/accel-sim/accel-sim-framework.git
```

{: .highlight}
> How to launch the container with ACCEL-SIM? Try to figure out by reading [LABI](LAB1.md)

To build ACCEL-SIM 
```
# in Docker, <CUDA_DIR>=/usr/local/cuda-11.0
export CUDA_INSTALL_PATH=<CUDA_DIR>
export PATH=$CUDA_INSTALL_PATH/bin:$PATH
source ./gpu-simulator/setup_environment.sh
make -j -C ./gpu-simulator/
```

To test your-built ACCEL-SIM 
```
. travis.sh
```

If you have problems about ACCEL-SIM, reference [its webpage](https://accel-sim.github.io/) first.

## A simple and interesting LAB about cache

At this part, we will add modification to the cache conifg and see what is the difference in the output.

{: .highlight}
> To simplify the LAB, we will give detailed steps to implement it.
> We have validated the following steps in docker container.

### Prepare benchmark

ACCEL-SIM provides a repo that collects gpu apps, to get it
```
git clone https://github.com/accel-sim/gpu-app-collection.git
```

There are many benchmark in it, we only need ispass-2009 at this LAB, to build it
```
# in Docker, <CUDA_DIR>=/usr/local/cuda-11.0
export CUDA_INSTALL_PATH=<CUDA_DIR>
export PATH=$CUDA_INSTALL_PATH/bin:$PATH
source ./src/setup_environment
make ispass-2009 -i -j -C ./src
make data -C ./src
```

To test the build
```
ls bin/<cuda-version>/release
```

{: .highlight}
> If you find `ispass-2009-BFS` and `ispass-2009-NN` in the output , then congratulate you build successfully.

### Modify some config files

ACCEL-SIM provide predefined benchmark and we only care about `ispass-2009-BFS` and `ispass-2009-NN`.
So add following content to `<accel-sim-framework>/util/job_launching/apps/define-all-apps.yml`
```
my-ispass-2009:
    exec_dir: "$GPUAPPS_ROOT/bin/$CUDA_VERSION/release/"
    data_dirs: "$GPUAPPS_ROOT/data_dirs/cuda/ispass-2009/"
    execs:
        - ispass-2009-BFS:
            - args:  ./data/graph65536.txt
        - ispass-2009-NN:
            - args:  28
```

ACCEL-SIM provide predefined configs, and we want to add a new config GTX480S which is a little different from GTX480.
To do so, add following content to `<accel-sim-framework>/util/job_launching/configs/define-standard-cfgs.yml`
```
GTX480S:
    base_file: "$GPGPUSIM_ROOT/configs/tested-cfgs/SM2_GTX480S/gpgpusim.config"
```

If you search `<accel-sim-framework>/gpu-simulator/gpgpu-sim/configs/tested-cfgs`, there isn't SM2_GTX480S.
So we create a dir `SM2_GTX480S`, and copy all the content under `SM2_GTX480` to `SM2_GTX480S`


At first, we didn't use the power simulation. So modify both `SM2_GTX480/gpgpusim.config` and `SM2_GTX480S/gpgpusim.config` with following content (diff format)
```
156,157c156,157
< -power_simulation_enabled 1
< -gpuwattch_xml_file gpuwattch_gtx480.xml
---
> -power_simulation_enabled 0
> # -gpuwattch_xml_file gpuwattch_gtx480.xml
```

Then we want to add some change to `SM2_GTX480S/gpgpusim.config` (diff format)
```
62c62
< -gpgpu_cache:dl1  N:32:128:4,L:L:m:N:H,S:64:8,8
---
> -gpgpu_cache:dl1  S:32:128:4,L:L:m:N:H,A:64:8,8
98c98
< -gpgpu_coalesce_arch 20
---
> -gpgpu_coalesce_arch 40
```

The main difference between `GTX480` and `GTX480S` is that at `GTX480` L1 cache is normal while 
at `GTX480S` L1 cache is sector.

{: .question}
> What is sector cache?

ACCEL-SIM provide predefined output colleciton configs,we want to generate output for 
`L2_total_cache_accesses` and `L1D_total_cache_miss_rate`.
So we add changes to `<accel-sim-framework>/util/job_launching/stats/example_stats.yml` (diff format)
```
41a42,43
>     - 'L2_total_cache_accesses\s+=\s*(.*)'
>     - 'L1D_total_cache_miss_rate\s+=\s*(.*)'
```

### Lanuch the simulation and collect output
  
to launch the simulation for config GTX480 and GTX480S
```
# at <accel-sim-framework> dir
./util/job_launching/run_simulations.py -C GTX480-PTX -B my-ispass-2009 -N test1
./util/job_launching/run_simulations.py -C GTX480S-PTX -B my-ispass-2009 -N test2
```

to monitor simulation
```
# at <accel-sim-framework> dir
./util/job_launching/monitor_func_test.py -N test1
./util/job_launching/monitor_func_test.py -N test2
```

to generate output 
```
# at <accel-sim-framework> dir
./util/job_launching/get_stats.py -B my-ispass-2009 -C GTX480-PTX,GTX480S-PTX | tee stats.csv
```

Try to take a look at the `stats.csv` and focus on `L2_total_cache_accesses` `L1D_total_cache_miss_rate` `gpu_tot_ipc` .

{: .question}
> What can you find from stats.csv?
> Why does BFS and NN vary differently on `gpu_tot_ipc`? Can you provide a detailed explanation?

{: .challenge}
> Can you explain why we make those changes at `gpgpusim.config` if we want to change the L1 cache to sector?

