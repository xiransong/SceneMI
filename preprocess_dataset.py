import argparse
import os

import numpy as np
import torch

from os.path import join as pjoin
from glob import glob

import trimesh
import pickle
import open3d as o3d

from tqdm import tqdm
import random
import copy

import smplx
import skimage.measure

from human_body_prior.tools.rotation_tools import aa2matrot, matrot2aa
from scipy.spatial.transform import Rotation 

from common.quaternion import *
from common.utils import point2point_signed

from utils_transform import *


def fixseed(seed):
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

device = "cpu"
base_dir = "/home/ubuntu/scratch/repos/SceneMI/datasets/TRUMANS/Data_release" # set your TRUMANS origin data here
smplx_model_path = "/home/ubuntu/scratch/repos/SceneMI/body_models"

fps = 30
motion_len = 121

K = 15
npoints = 1024

seed = 42
fixseed(seed)

scene_mesh_dir = 'Scene_mesh'
scene_npy_dir = 'Scene'
object_mesh_dir = 'Object_all/Object_mesh'
object_pcd_dir = 'Object_all/Object_pcd'
object_pose_dir = 'Object_all/Object_pose'

scale_x, scale_y, scale_z = 300., 100., 400.
scale = 100.
sdf_levels = [0.0]

noise_levels = [0.0, 0.2, 0.5, 1.0]

face_joint_indx = [2, 1, 17, 16]
r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
uniform_verts_idx = []
all_verts_idx = []


sbj_m_bs = smplx.create(model_path=smplx_model_path,
                model_type='smplx',
                gender="neutral",
                use_pca=False,
                flat_hand_mean=True,
                batch_size=motion_len).to(device).eval()

verts_id_path = 'smplx_verts_id_uniform_ds_{}.pt'.format(K)

if os.path.exists(verts_id_path):
    print("get verts id from {}".format(verts_id_path))
    uniform_verts_idx = torch.load(verts_id_path).tolist()

else:
    print("get downsample verts id")
    sbj_m_single = smplx.create(model_path=smplx_model_path,
                model_type='smplx',
                gender="neutral",
                use_pca=False,
                flat_hand_mean=True,
                batch_size=1).to(device).eval()
    
    tpose_body_params = {'body_pose': torch.zeros(1,63), 'transl': torch.zeros(1,3), 
                        'global_orient': torch.zeros(1,3)}
    tpose_sbj = sbj_m_single(**tpose_body_params)
    tpose_verts = tpose_sbj.vertices.reshape(-1,3).detach().cpu().numpy()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(tpose_verts.reshape(-1,3))
    #o3d.io.write_point_cloud(os.path.join("tpose.pcd"), pcd)
    sub_pcd = pcd.farthest_point_down_sample(tpose_verts.reshape(-1,3).shape[0]//K)
    #o3d.io.write_point_cloud(os.path.join("sub_tpose.pcd"), sub_pcd)

    pcd_points = np.asarray(pcd.points)
    for points in np.asarray(sub_pcd.points):
        v_idx = np.linalg.norm(np.abs(pcd_points - points.reshape(-1,3)),axis=1).argmin()
        uniform_verts_idx.append(v_idx)

    with open(verts_id_path, "wb") as f:
        torch.save(torch.tensor(uniform_verts_idx), f)


sbj_m_single = smplx.create(model_path=smplx_model_path,
                model_type='smplx',
                gender="neutral",
                use_pca=False,
                flat_hand_mean=True,
                batch_size=1).to(device).eval()


def add_synthetic_noise(body_params, noise_level=1.):
    
    noise_level_cm = noise_level / 100.           # scale at cm
    noise_level_deg = noise_level * (np.pi/180.)  # scale at deg

    body_pose = body_params['body_pose'].clone()
    transl = body_params['transl'].clone()
    global_orient =  body_params['global_orient'].clone()

    if 'betas' in body_params.keys():
        betas = body_params['betas'].clone()

    n_add_noise = transl.shape[0] - 2
    
    body_pose_noise = torch.randn(n_add_noise, body_pose.shape[1]) * noise_level_deg
    transl_noise = torch.randn(n_add_noise, transl.shape[1]) * noise_level_cm
    global_orient_noise = torch.randn(n_add_noise, global_orient.shape[1]) * noise_level_deg

    body_pose[1:-1,:] += body_pose_noise
    transl[1:-1,:] += transl_noise
    global_orient[1:-1,:] += global_orient_noise

    noisy_body_params = {'body_pose': body_pose, 
                        'transl': transl, 
                        'global_orient': global_orient,
                        'body_pose_6d': aa2six_rand(body_pose),
                        'global_orient_6d': aa2six_rand(global_orient)}

    if 'betas' in body_params.keys():
        noisy_body_params['betas'] = body_params['betas'].clone()
    
    return noisy_body_params


def canonicalized(body_params, scene, scene_points):

    body_pose = body_params['body_pose'].clone()
    transl = body_params['transl'].clone()
    global_orient = body_params['global_orient'].clone()
    #betas = body_params['betas'].clone()

    scene_offset = np.array((transl[0][0], 0.0, transl[0][2]))

    scene.vertices -= np.array((transl[0][0], 0.0, transl[0][2])) 
    scene_points -= np.array((transl[0][0], 0.0, transl[0][2])) 

    transl -= np.array((transl[0][0], 0.0, transl[0][2]))
    
    
    bm_output = sbj_m_bs(**body_params)

    root_pos_init = np.asarray(bm_output.joints.detach().cpu().numpy()[0, :22, :])
    
    across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
    across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
    across = across1 + across2
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]

    # forward (3,), rotate around y-axis
    forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
    # forward (3,)
    forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]

    target = np.array([[0, 0, 1]])
    root_quat_init = qbetween_np(forward_init, target)
    root_matrot_init = quaternion_to_matrix(torch.tensor(root_quat_init))
    root_euler_init = qeuler(torch.tensor(root_quat_init), order="zyx", epsilon=0, deg=False)
    root_aa_init = matrot2aa(quaternion_to_matrix(torch.tensor(root_quat_init)))

    root_quat_init_inv = qbetween_np(target, forward_init)
    root_aa_init_inv = matrot2aa(quaternion_to_matrix(torch.tensor(root_quat_init_inv)))

    #bdata_poses_y = global_orient[:, :3].copy()
    #bdata_poses_y[:,1] = 0.0
    bdata_root_rotmat = aa2matrot(torch.tensor(global_orient[:, :3]))

    bdata_poses_offset = root_aa_init.numpy()

    cano_rotmat = aa2matrot(torch.tensor(bdata_poses_offset.reshape(1,3)))[0].float()
    inv_cano_rotmat = aa2matrot(torch.tensor(-bdata_poses_offset.reshape(1,3)))[0].float()

    bdata_root_rotmat_cano = torch.matmul(bdata_root_rotmat.float(), cano_rotmat)
    bdata_root_cano = matrot2aa(bdata_root_rotmat_cano)

    bdata_poses_matrot = aa2matrot(torch.tensor(global_orient[:, :3])).float()
    global_orient[:, :3] = matrot2aa(torch.matmul(root_matrot_init[0], bdata_poses_matrot.T).T)
    transl[:] = torch.matmul(root_matrot_init[0], torch.tensor(transl[:]).float().T).T

    scene.vertices = torch.matmul(root_matrot_init[0], torch.tensor(scene.vertices[:]).float().T).T
    scene_points = torch.matmul(root_matrot_init[0], torch.tensor(scene_points[:]).float().T).T

    scene_rotmat = root_matrot_init[0]

    cano_body_params = {'body_pose': body_pose, 'transl': transl, 'global_orient': global_orient,
                        'body_pose_6d': aa2six_rand(body_pose),
                        'global_orient_6d': aa2six_rand(global_orient)
                        }

    if 'betas' in body_params.keys():
        cano_body_params['betas'] = body_params['betas'].clone()

    if not 'global_joints' in cano_body_params.keys():
        bm_output = sbj_m_bs(**cano_body_params)
        body_params['global_joints'] = bm_output.joints[:,:24,:]
    
    bm_verts = bm_output.vertices

    verts_min_max_xz = np.array([bm_verts[:,:,0].min().item(), bm_verts[:,:,0].max().item(), bm_verts[:,:,2].min().item(), bm_verts[:,:,2].max().item()])

    return cano_body_params, scene, scene_points, scene_offset, scene_rotmat, verts_min_max_xz


def main(args):

    out_base_dir = "preprocess_{}_beta{}".\
                        format(motion_len-1, args.beta)

    print(out_base_dir)

    frame_id = np.load(pjoin(base_dir, 'frame_id.npy'))

    human_pose = torch.tensor(np.load(pjoin(base_dir, 'human_pose.npy')))
    human_orient = torch.tensor(np.load(pjoin(base_dir, 'human_orient.npy')))
    human_transl = torch.tensor(np.load(pjoin(base_dir, 'human_transl.npy')))
    human_joints = torch.tensor(np.load(pjoin(base_dir, 'human_joints.npy')))
    human_betas = torch.tensor(np.load(pjoin(base_dir, 'betas.npy')))

    seg_name = np.load(pjoin(base_dir, 'seg_name.npy'))
    frame_id = np.load(pjoin(base_dir, 'frame_id.npy'))
    scene_list = np.load(pjoin(base_dir, 'scene_list.npy'))
    scene_flag = np.load(pjoin(base_dir, 'scene_flag.npy'))
    object_list = np.load(pjoin(base_dir, 'object_list.npy'))
    object_flag = np.load(pjoin(base_dir, 'object_flag.npy'))
    object_mat = np.load(pjoin(base_dir, 'object_mat-001.npy'))

    meta = np.load(pjoin(base_dir, 'meta.npy'), allow_pickle=True)
    joint_id = meta.item()['joints_ind']
    action_label = meta.item()['action_label']

    norm = np.load(pjoin(base_dir, 'norm.npy'), allow_pickle=True)
    idx_start = np.load(pjoin(base_dir, 'idx_start.npy'), allow_pickle=True)
    bad_frames = torch.tensor(np.load(pjoin(base_dir, 'bad_frames.npy')))

    scene_size = 'xz24'
    save_scene = False

    grid_res_xz = 48
    grid_res_y = 24

    grid_max_xz = 2.4
    grid_max_y = 1.2

    axis_xz = torch.linspace(-grid_max_xz, grid_max_xz, grid_res_xz, device = 'cuda')
    axis_y = torch.linspace(-grid_max_y, grid_max_y, grid_res_y, device = 'cuda')
    p_x, p_y, p_z = torch.meshgrid(axis_xz, axis_y + grid_max_y, axis_xz)

    points = torch.cat((p_x.unsqueeze(-1), p_y.unsqueeze(-1), p_z.unsqueeze(-1)), dim=3)
    points = points.view(1, -1, 3)
    points_y_idx = torch.range(1, grid_res_y).unsqueeze(1).unsqueeze(0).repeat(grid_res_xz, 1, grid_res_xz).cuda()

    
    object_pcd_path = pjoin(base_dir, object_pcd_dir)

    if not os.path.exists(object_pcd_path):
        import kaolin as kal
        os.makedirs(object_pcd_path)

        uni_axis = torch.linspace(-0.5, 0.5, 32, device = 'cuda')
        uni_p_x, uni_p_y, uni_p_z = torch.meshgrid(uni_axis, uni_axis, uni_axis)
        uni_points_origin = torch.cat((uni_p_x.unsqueeze(-1), uni_p_y.unsqueeze(-1), uni_p_z.unsqueeze(-1)), dim=3)

        for _, obj_mesh_file in enumerate(sorted(glob(pjoin(base_dir, object_mesh_dir, "*.obj")))):
            obj_mesh = trimesh.load(obj_mesh_file)
            min_x, min_y, min_z = torch.tensor(obj_mesh.vertices.min(axis=0))
            max_x, max_y, max_z = torch.tensor(obj_mesh.vertices.max(axis=0))

            uni_points = uni_points_origin.clone().view(1, -1, 3)

            uni_points[:,:,0] *= (max_x - min_x)
            uni_points[:,:,1] *= (max_y - min_y)
            uni_points[:,:,2] *= (max_z - min_z)

            uni_points[:,:,0] += (max_x + min_x) / 2
            uni_points[:,:,1] += (max_y + min_y) / 2
            uni_points[:,:,2] += (max_z + min_z) / 2

            verts = torch.tensor(np.asarray(obj_mesh.vertices), device = 'cuda').unsqueeze(0)
            faces = torch.tensor(np.asarray(obj_mesh.faces), device = 'cuda')

            points_sign = kal.ops.mesh.check_sign(verts, faces, uni_points)

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.array(uni_points[:,torch.where(points_sign == True)[1]].detach().cpu()).reshape(-1,3))
            
            out_path = pjoin(object_pcd_path, os.path.split(obj_mesh_file)[1][:-4]+'.pcd')
            o3d.io.write_point_cloud(str(out_path), pcd)

    seg_list = sorted(np.unique(seg_name))

    for _, seg in enumerate(tqdm(seg_list)):

        seg_preprocess_folder = pjoin(out_base_dir, seg)
        os.makedirs(seg_preprocess_folder, exist_ok=True)

        seg_idx = np.where(seg_name == seg)[0]
        bs = seg_idx.shape[0]

        body_pose = human_pose[seg_idx]
        transl = human_transl[seg_idx]
        global_orient = human_orient[seg_idx]
        body_joints = human_joints[seg_idx]
        betas = human_betas[seg_idx]
        
        assert np.unique(scene_flag[seg_idx]).shape[0] == 1

        scene_idx = scene_flag[seg_idx][0]
        scene_name = scene_list[scene_idx]

        scene_mesh_path = pjoin(base_dir, scene_mesh_dir, scene_name + ".obj")
        scene_npy_path = pjoin(base_dir, scene_npy_dir, scene_name + ".npy")

        if not os.path.exists(scene_mesh_path) or not os.path.exists(scene_npy_path):
            continue

        scene_mesh = trimesh.load(scene_mesh_path)
        
        scene_npy = np.load(scene_npy_path)

        vtx, faces, _, _ = skimage.measure.marching_cubes(scene_npy, 0.0)

        vtx[:,0] *= scale_x / (scale_x-1)
        vtx[:,1] *= scale_y / (scale_y-1)
        vtx[:,2] *= scale_z / (scale_z-1)

        vtx /= (scale/2.)

        vtx[:,0] -= scale_x / scale
        vtx[:,2] -= scale_z / scale

        scene_mesh_recon = trimesh.Trimesh(vtx, faces)

        scene_xx, scene_yy, scene_zz = np.where(scene_npy > 0)

        scene_xx = np.array(scene_xx, dtype=np.float64)
        scene_yy = np.array(scene_yy, dtype=np.float64)
        scene_zz = np.array(scene_zz, dtype=np.float64)

        scene_xx *= scale_x / (scale_x-1)
        scene_yy *= scale_y / (scale_y-1)
        scene_zz *= scale_z / (scale_z-1)

        scene_xx /= (scale/2.)
        scene_yy /= (scale/2.)
        scene_zz /= (scale/2.)

        scene_xx -= scale_x / scale
        scene_zz -= scale_z / scale

        scene_pcd_origin = o3d.geometry.PointCloud()
        scene_pcd_origin.points = o3d.utility.Vector3dVector(np.stack([scene_xx,scene_yy,scene_zz]).T)

        object_pose_npy_path = pjoin(base_dir, object_pose_dir, seg[:19] + ".npy")
        if not os.path.exists(object_pose_npy_path):
            continue

        object_pose_npy = np.load(object_pose_npy_path, allow_pickle=True).item()
        object_list = []
        object_list_pcd = []
        object_transformations = []

        for _, key in enumerate(object_pose_npy.keys()):
            object_list.append(trimesh.load(pjoin(base_dir, object_mesh_dir, key + ".obj")))
            object_list_pcd.append(o3d.io.read_point_cloud(pjoin(base_dir, object_pcd_dir, key + ".pcd")))

            object_transformation = torch.zeros((bs, 4, 4))
            object_transformation[:,:3,:3] = torch.tensor(Rotation.from_euler("xyz",torch.tensor(np.array(object_pose_npy[key]['rotation'])),degrees=False).as_matrix())
            object_transformation[:,:3,3] = torch.tensor(np.array(object_pose_npy[key]['location']))
            object_transformation[:,3,3] = 1.0
            object_transformations.append(object_transformation)

        object_list_origin = object_list.copy()

        scene_mesh_origin = scene_mesh

        split_seg_idx = torch.split(torch.tensor(seg_idx), motion_len)
        split_local_seg_idx = torch.split(torch.tensor(seg_idx) - seg_idx.min(), motion_len)

        for _seg_num, _seg_idx in enumerate(split_seg_idx):
            scene_mesh = scene_mesh_origin.copy()
            scene_pcd = copy.deepcopy(scene_pcd_origin)
            
            if _seg_idx.shape[0] == motion_len:
                if np.intersect1d(_seg_idx, bad_frames).shape[0] == 0:

                    save_dir = pjoin(seg_preprocess_folder, "{}".format(_seg_idx[0]))
                    os.makedirs(save_dir, exist_ok=True)

                    object_list = object_list_origin.copy()
                    for _idx in range(len(object_list)):
                        object_list[_idx].apply_transform(object_transformations[_idx][split_local_seg_idx[_seg_num][0]])
                        object_pcd_t = copy.deepcopy(object_list_pcd[_idx]).transform(object_transformations[_idx][split_local_seg_idx[_seg_num][0]])

                        scene_mesh += object_list[_idx]
                        scene_pcd += object_pcd_t
                
                    if save_scene:
                        scene_mesh.export(pjoin(save_dir, "scene.obj"))
                        o3d.io.write_point_cloud(str(pjoin(save_dir, "scene.pcd")), scene_pcd)

                    body_params_gt = {'body_pose': body_pose[split_local_seg_idx[_seg_num]], 
                                'body_pose_6d': aa2six_rand(body_pose[split_local_seg_idx[_seg_num]]),
                                'transl': transl[split_local_seg_idx[_seg_num]], 
                                'global_orient': global_orient[split_local_seg_idx[_seg_num]],
                                'global_orient_6d': aa2six_rand(global_orient[split_local_seg_idx[_seg_num]]),
                                }

                    if args.beta: 
                        body_params_gt['betas'] = betas[split_local_seg_idx[_seg_num]]
                        # pelvis_to_head, shoulder, arm, hip, leg, hip_depth, chest_depth
                        # pre-calculated; please refer to sample.conditional_synthesis python file
                        if body_params_gt['betas'][0][0] > 0.7:
                            height = 1.7326
                            part_height = np.array([0.6309418, 0.092184424, 0.50035673, 0.11127687, 0.779747, 0.26611862, 0.23075028])

                        elif body_params_gt['betas'][0][0] > 0.5:
                            height = 1.7387
                            part_height = np.array([0.6636606, 0.09084908, 0.5078993, 0.11199133, 0.7500247, 0.26906574, 0.23780064])

                        elif body_params_gt['betas'][0][0] > -0.5:
                            height = 1.7933
                            part_height = np.array([0.6702256, 0.09400587, 0.528662, 0.116085745, 0.7926093, 0.2735553, 0.23692235])

                        elif body_params_gt['betas'][0][0] > -1.3:
                            height = 1.8618
                            part_height = np.array([0.6951775, 0.09901273, 0.5503893, 0.11992166, 0.8296785, 0.27713653, 0.24794094])

                        body_params_gt['part_height'] = part_height
                    
                    

                    with open(pjoin(save_dir, 'body_parms_gt.pickle'),'wb') as fw:
                        pickle.dump(body_params_gt, fw)

                    body_params_cano, scene_cano, scene_cano_pcd, scene_offset, scene_rotmat, verts_min_max_xz = canonicalized(body_params_gt, scene_mesh, torch.tensor(np.asarray(scene_pcd.points)))
                    scene_info = {'scene_offset': torch.tensor(scene_offset),
                                'scene_rotmat': scene_rotmat,
                                'scene_mesh_path': scene_mesh_path,
                                'seg_idx': _seg_idx,
                                'verts_min_max_xz': verts_min_max_xz}

                    with open(pjoin(save_dir, 'scene_info.pickle'),'wb') as fw:
                        pickle.dump(scene_info , fw)

                    if save_scene:
                        scene_cano.export(pjoin(save_dir, "scene_cano.obj"))
                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(np.array(scene_cano_pcd).reshape(-1,3))
                        o3d.io.write_point_cloud(str(pjoin(save_dir, "scene_cano.pcd")), pcd)

                    if args.occ_scene:
                        _, points_dist = point2point_signed(points.float(), scene_cano_pcd.unsqueeze(0).cuda())
                        dist_threshold = (2. * grid_max_y / grid_res_y) / 2.
                        is_occupied = torch.norm(points_dist, dim=2)[0] < dist_threshold
                        points_sign = is_occupied.reshape((grid_res_xz, grid_res_y, grid_res_xz))

                        #np.save(pjoin(save_dir, 'occ_scene.npy'), points_sign.detach().cpu().numpy())
                        np.savez(pjoin(save_dir, 'occ_scene.npz'), occ_scene=points_sign.detach().cpu().numpy())

                    with open(pjoin(save_dir, 'body_parms_cano_gt.pickle'),'wb') as fw:
                        pickle.dump(body_params_cano, fw)

                    for noise_level in noise_levels:
                        body_params_cano_noisy = add_synthetic_noise(body_params_cano, noise_level=noise_level)
                        with open(pjoin(save_dir, 'body_parms_cano_noisy_{}.pickle'.format(noise_level)),'wb') as fw:
                            pickle.dump(body_params_cano_noisy, fw)

                        verts = torch.tensor(np.asarray(scene_mesh.vertices), device = 'cuda').unsqueeze(0)
                        faces = torch.tensor(np.asarray(scene_mesh.faces), device = 'cuda')

                        sbj_verts = sbj_m_bs(**body_params_cano_noisy).vertices[:, uniform_verts_idx]
                        _, bps_sbj = point2point_signed(sbj_verts.reshape(1,-1,3).cuda(), verts.float())
                        bps_sbj = bps_sbj.reshape(sbj_verts.shape)
                        np.save(pjoin(save_dir, 'bps_sbj_{}.npy'.format(noise_level)), bps_sbj.detach().cpu().numpy())


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Data Preprocessing"
    )
    parser.add_argument('--beta', action='store_true', default=False, help='If set, consider body shape while processing')
    parser.add_argument('--occ_scene', action='store_true', default=False, help='If set, save occupancy for each segment')
    
    args = parser.parse_args()

    main(args)