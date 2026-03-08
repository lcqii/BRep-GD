#!/bin/bash

# Adjust the timeout time accordingly, memory could explode when loading few very large cad models


### Process DeepCAD data ###
# for i in $(seq 0 18)
# do
#     # Call python script with different interval values
#     timeout 500 python process_brep.py --input /path/to/abc/unzip_step --interval $i --option 'deepcad'
#     pkill -9 -u luochenqi -f '^python process_brep.py' # cleanup after each run
# done


# ### Process ABC data ###
# for i in $(seq 0 99)
# do
#     # Call python script with different interval values
#     timeout 1000 python process_brep.py --input path/to/your/abc_step --interval $i --option 'abc'
#     pkill -f '^python process_brep.py' # cleanup after each run
# done


# ### Process Furniture data ###
# python process_brep.py --input /path/to/datasets/furniture_dataset --option 'furniture'


### Process Furniture data ###
python process_brep.py --input /path/to/CADNet50/cadnet40v2_STEP --option 'cadnet40v2'