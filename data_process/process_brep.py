import os 
import pickle 
import argparse
from regex import P
from tqdm import tqdm
from multiprocessing.pool import Pool
from convert_utils import *
from occwl.io import load_step
import gc

MAX_FACE = 70

def normalize(surf_pnts, edge_pnts, corner_pnts):
    """
    Various levels of normalization 
    """
    total_points = np.array(surf_pnts).reshape(-1, 3)
    min_vals = np.min(total_points, axis=0)
    max_vals = np.max(total_points, axis=0)
    global_offset = min_vals + (max_vals - min_vals)/2 
    global_scale = max(max_vals - min_vals)
    assert global_scale != 0, 'scale is zero'

    surfs_wcs, edges_wcs, surfs_ncs, edges_ncs = [],[],[],[]

    corner_wcs = (corner_pnts - global_offset[np.newaxis,:]) / (global_scale * 0.5)

    for surf_pnt in surf_pnts:    
        surf_pnt_wcs = (surf_pnt - global_offset[np.newaxis,np.newaxis,:]) / (global_scale * 0.5)
        surfs_wcs.append(surf_pnt_wcs)
        min_vals = np.min(surf_pnt_wcs.reshape(-1,3), axis=0)
        max_vals = np.max(surf_pnt_wcs.reshape(-1,3), axis=0)
        local_offset = min_vals + (max_vals - min_vals)/2 
        local_scale = max(max_vals - min_vals)
        pnt_ncs = (surf_pnt_wcs - local_offset[np.newaxis,np.newaxis,:]) / (local_scale * 0.5)
        surfs_ncs.append(pnt_ncs)
       
    for edge_pnt in edge_pnts:    
        edge_pnt_wcs = (edge_pnt - global_offset[np.newaxis,:]) / (global_scale * 0.5)
        edges_wcs.append(edge_pnt_wcs)
        min_vals = np.min(edge_pnt_wcs.reshape(-1,3), axis=0)
        max_vals = np.max(edge_pnt_wcs.reshape(-1,3), axis=0)
        local_offset = min_vals + (max_vals - min_vals)/2 
        local_scale = max(max_vals - min_vals)
        pnt_ncs = (edge_pnt_wcs - local_offset) / (local_scale * 0.5)
        edges_ncs.append(pnt_ncs)
        assert local_scale != 0, 'scale is zero'

    surfs_wcs = np.stack(surfs_wcs)
    surfs_ncs = np.stack(surfs_ncs)
    edges_wcs = np.stack(edges_wcs)
    edges_ncs = np.stack(edges_ncs)

    return surfs_wcs, edges_wcs, surfs_ncs, edges_ncs, corner_wcs


def parse_solid(solid):
    """
    Parse the surface, curve, face, edge, vertex in a CAD solid.
   
    Args:
    - solid (occwl.solid): A single brep solid in occwl data format.

    Returns:
    - data: A dictionary containing all parsed data
    """
    assert isinstance(solid, Solid),"solid type wrong (parse_solid)"

    solid = solid.split_all_closed_faces(num_splits=0)
    solid = solid.split_all_closed_edges(num_splits=0)
    
    solid=CurvsConvertToBezier(solid)
    solid=unify_same_domain(solid)
    solid=Solid(solid)

    if len(list(solid.faces())) > MAX_FACE:
        return None
        
    face_pnts, edge_pnts, edge_corner_pnts, edgeFace_IncM, faceEdge_IncM = extract_primitive(solid)
    
    surfs_wcs, edges_wcs, surfs_ncs, edges_ncs, corner_wcs = normalize(face_pnts, edge_pnts, edge_corner_pnts)

    corner_wcs = np.round(corner_wcs,4) 
    corner_unique = []
    for corner_pnt in corner_wcs.reshape(-1,3):
        if len(corner_unique) == 0:
            corner_unique = corner_pnt.reshape(1,3)
        else:
            exists = np.any(np.all(corner_unique == corner_pnt, axis=1))
            if exists:
                continue 
            else:
                corner_unique = np.concatenate([corner_unique, corner_pnt.reshape(1,3)], 0)

    edgeCorner_IncM = []
    for edge_corner in corner_wcs:
        start_corner_idx = np.where((corner_unique == edge_corner[0]).all(axis=1))[0].item()
        end_corner_idx = np.where((corner_unique == edge_corner[1]).all(axis=1))[0].item()
        edgeCorner_IncM.append([start_corner_idx, end_corner_idx])
    edgeCorner_IncM = np.array(edgeCorner_IncM)

    surf_bboxes = []
    for pnts in surfs_wcs:
        min_point, max_point = get_bbox(pnts.reshape(-1,3))
        surf_bboxes.append( np.concatenate([min_point, max_point]))
    surf_bboxes = np.vstack(surf_bboxes)

    edge_bboxes = []
    for pnts in edges_wcs:
        min_point, max_point = get_bbox(pnts.reshape(-1,3))
        edge_bboxes.append(np.concatenate([min_point, max_point]))
    edge_bboxes = np.vstack(edge_bboxes)

    data = {
        'surf_wcs':surfs_wcs.astype(np.float32),
        'edge_wcs':edges_wcs.astype(np.float32),
        'surf_ncs':surfs_ncs.astype(np.float32),
        'edge_ncs':edges_ncs.astype(np.float32),
        'corner_wcs':corner_wcs.astype(np.float32),
        'edgeFace_adj': edgeFace_IncM,
        'edgeCorner_adj':edgeCorner_IncM,
        'faceEdge_adj':faceEdge_IncM,
        'surf_bbox_wcs':surf_bboxes.astype(np.float32),
        'edge_bbox_wcs':edge_bboxes.astype(np.float32),
        'corner_unique':corner_unique.astype(np.float32),
    }

    return data


def process(step_folder):
    """
    Helper function to load step files and process in parallel

    Args:
    - step_folder (str): Path to the STEP parent folder.

    Returns:
    - Complete status: Valid (1) / Non-valid (0).
    """

        





    try:

        process_cad=False
        if step_folder.endswith('.step'):
            if 'CADNet' in step_folder:
                step_path = step_folder 
                process_furniture = True
                process_cad = True
            else:   
                step_path = step_folder 
                process_furniture = True
        else:
            for _, _, files in os.walk(step_folder):
                assert len(files) == 1 
                step_path = os.path.join(step_folder, files[0])
            process_furniture = False

        cad_solid = load_step(step_path)
        if len(cad_solid)!=1: 
            return -1 
            
        data = parse_solid(cad_solid[0])
        if data is None: 
            return -2 # number of faces or edges exceed pre-determined threshold

        if process_cad:
            data_uid = step_path.split('/')[-2] + '_' + step_path.split('/')[-1]
            sub_folder = step_path.split('/')[-2]
        elif process_furniture:
            data_uid = step_path.split('/')[-2] + '_' + step_path.split('/')[-1]
            sub_folder = step_path.split('/')[-3]
        else:
            data_uid = step_path.split('/')[-2]
            sub_folder = data_uid[:4]

        data['uid'] = data_uid
        save_folder = os.path.join(OUTPUT, sub_folder)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder,exist_ok=True)

        save_path = os.path.join(save_folder, data['uid']+'.pkl')
        with open(save_path, "wb") as tf:
            pickle.dump(data, tf)

        return 1 

            

            




    except Exception as e:

        print('not saving due to error...'+step_folder)
        print(str(e))

        return 0
def worker_with_timeout(step_dir, timeout=300):
    try:
        # 创建一个 Pool 来执行任务
        with Pool(1) as pool:  
            result = pool.apply_async(process, (step_dir,))
            # 如果超时，抛出异常并返回 2
            return result.get(timeout=timeout)
    except Exception as e:
        # 超时或者其他异常，返回 2 表示超时
        return 2  # 超时任务返回 2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Data folder path", required=True)
    parser.add_argument("--option", type=str, choices=['abc', 'deepcad', 'furniture','CADNet50','CADNet40','CADNet30',], default='abc', 
                        help="Choose between dataset option [abc/deepcad/furniture] (default: abc)")
    parser.add_argument("--interval", type=int, help="Data range index, only required for abc/deepcad")
    args = parser.parse_args()    

    if args.option == 'deepcad': 
        OUTPUT = 'deepcad_parsed'
    elif args.option == 'abc': 
        OUTPUT = 'abc_parsed'
    elif args.option == 'CADNet50': 
        OUTPUT = 'cadnet40v2_parsed'
    elif args.option == 'CADNet40': 
        OUTPUT = 'CADNet40_parsed'
    elif args.option == 'CADNet30': 
        OUTPUT = 'CADNet30_parsed'
    else:
        OUTPUT = 'furniture_parsed'
    
    if args.option in ['furniture','CADNet50','CADNet40','CADNet30']:
        step_dirs = load_furniture_step(args.input)
    else:
        step_dirs = load_abc_step(args.input, args.option=='deepcad')
        step_dirs = step_dirs[args.interval*10000 : (args.interval+1)*10000]

    valid = 0
    multi_solids=0
    Exceeding_threshold=0
    err=0

    timeout_count = 0  # 用于统计超时的任务数量
    timeout = 150  # 设置超时时间阈值，单位秒

    # 创建进程池，使用所有 CPU 核心进行并行处理
    with Pool(os.cpu_count()) as pool:
        convert_iter = (pool.apply_async(process, args=(step_dir,)) for step_dir in step_dirs)

        
        # 使用 tqdm 显示处理进度
        with tqdm(total=len(step_dirs)) as tqdm_iter:
            for status in convert_iter:
                try:
                    result = status.get(timeout=timeout)  # 等待每个任务的结果
                    if result == 0:
                        err += 1
                    elif result == -1:
                        multi_solids += 1
                    elif result == -2:
                        Exceeding_threshold += 1
                    elif result == 1:  # 任务合法
                        valid += 1
                    # elif result == 2:  # 超时
                except Exception as e:
                    timeout_count += 1
                    #print(f"处理任务时出错: {str(e)}")

                # 动态更新进度条的描述
                tqdm_iter.set_description(f"valid {valid} | time out {timeout_count}|multi {multi_solids}|thr {Exceeding_threshold}|err {err}")
                tqdm_iter.update(1)

    print(f"有效处理的任务数量: {valid}")
    print(f"超时的任务数量: {timeout_count}")


    # # 单线程 单状态


    # # 多线程并行 批次 多状态
    # batch_size = 1000  # 每批处理1000个文件
        
        
    #     # 清理当前批次的资源




    # #多线程并行 单状态


    # #多线程并行 多状态
        


