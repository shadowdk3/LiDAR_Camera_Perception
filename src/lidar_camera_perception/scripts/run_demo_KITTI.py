import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET

def load_kitti_calib(calib_dir):
    """讀取 KITTI 光達與相機校正矩陣"""
    # 這裡以預設的標準矩陣或從 calib 檔案讀取為例
    velo_to_cam0_ext = np.array([
        [ 7.533745e-03, -9.999714e-01, -6.166020e-04, -4.069766e-03],
        [ 1.480249e-02,  7.280733e-04, -9.998902e-01, -7.631618e-02],
        [ 9.998621e-01,  7.523790e-03,  1.480755e-02, -2.717806e-01],
        [ 0.000000e+00,  0.000000e+00,  0.000000e+00,  1.000000e+00]
    ])
    cam0_rectification = np.array([
        [ 9.999239e-01,  9.837760e-03, -7.445048e-03,  0.000000e+00],
        [-9.869795e-03,  9.999421e-01, -4.278459e-03,  0.000000e+00],
        [ 7.402527e-03,  4.351614e-03,  9.999631e-01,  0.000000e+00],
        [ 0.000000e+00,  0.000000e+00,  0.000000e+00,  1.000000e+00]
    ])
    cam2_projection_rectified = np.array([
        [7.215377e+02, 0.000000e+00, 6.095593e+02, 4.485728e+01],
        [0.000000e+00, 7.215377e+02, 1.728540e+02, 2.163791e-01],
        [0.000000e+00, 0.000000e+00, 1.000000e+00, 2.745884e-03]
    ])
    return cam2_projection_rectified @ cam0_rectification @ velo_to_cam0_ext

def parse_tracklets(xml_path, target_frame=72):
    """解析 XML Tracklet 檔案（加入安全檢查，避免抓取不到標籤時崩潰）"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    tracklets = []
    
    items = root.findall(".//item")
    print(f"number of items: {len(items)}")
    
    for idx, item in enumerate(root.findall('.//item')):
        # 安全取得 objectType
        obj_type_node = item.find('objectType')
        obj_type = obj_type_node.text if obj_type_node is not None else 'Unknown'
        
        # 安全取得尺寸數據
        h_node = item.find('h')
        w_node = item.find('w')
        l_node = item.find('l')
        ff_node = item.find('first_frame')
        
        tracklet = {
            'objectType': obj_type,
            'h': float(h_node.text) if h_node is not None else 0.0,
            'w': float(w_node.text) if w_node is not None else 0.0,
            'l': float(l_node.text) if l_node is not None else 0.0,
            'first_frame': int(ff_node.text) if ff_node is not None else 0,
            'poses': []
        }
        
        poses_node = item.find('poses')
        if poses_node is not None:
            for pose_item in poses_node.findall('item'):
                tx_node = pose_item.find('tx')
                ty_node = pose_item.find('ty')
                tz_node = pose_item.find('tz')
                rz_node = pose_item.find('rz')
                
                tracklet['poses'].append({
                    'tx': float(tx_node.text) if tx_node is not None else 0.0,
                    'ty': float(ty_node.text) if ty_node is not None else 0.0,
                    'tz': float(tz_node.text) if tz_node is not None else 0.0,
                    'rz': float(rz_node.text) if rz_node is not None else 0.0
                })
        tracklets.append(tracklet)
        
        # 檢查該 tracklet 是否涵蓋指定的影格 (72)
        # pose_idx = target_frame - tracklet['first_frame']
        
        # if 0 <= pose_idx < len(tracklet['poses']):
        #     p = tracklet['poses'][pose_idx]
        #     print(f"Tracklet Index: {idx}")
        #     print(f"  - Object Type: {obj_type}")
        #     print(f"  - Dimensions : h={tracklet['h']:.3f}, w={tracklet['w']:.3f}, l={tracklet['l']:.3f}")
        #     print(f"  - First Frame: {tracklet['first_frame']} (Current Pose Index: {pose_idx})")
        #     print(f"  - 3D Position & Rotation:")
        #     print(f"      tx: {p['tx']:.4f}")
        #     print(f"      ty: {p['ty']:.4f}")
        #     print(f"      tz: {p['tz']:.4f}")
        #     print(f"      rz: {p['rz']:.4f}")
        #     print("-" * 40)
            
    return tracklets

def run_visualization(base_dir):
    image_dir = os.path.join(base_dir, "image_00", "data")
    # image_dir = os.path.join(base_dir, "image_02", "data")
    xml_path = os.path.join(base_dir, "tracklet_labels.xml")
    
    tracklets = parse_tracklets(xml_path)
    projection_matrix = load_kitti_calib(base_dir)
    
    images = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
    
    print(f"Number of tracklets: {len(tracklets)}")
    print(f"Number of images: {len(images)}")
    for img_idx, img_name in enumerate(images):
        img_path = os.path.join(image_dir, img_name)
        img = cv2.imread(img_path)
        
        for tracklet in tracklets:
            start_frame = tracklet['first_frame']
            pose_idx = img_idx - start_frame
            
            if '0000000072' in img_name:
                print_info = True
                print("img_idx", img_idx)
                print("start_frame", start_frame)
                print("pose_idx", pose_idx)
                print("pose", len(tracklet['poses']))
            
            if 0 <= pose_idx < len(tracklet['poses']):
                pose = tracklet['poses'][pose_idx]
                h, w, l = tracklet['h'], tracklet['w'], tracklet['l']
                tx, ty, tz, rz = pose['tx'], pose['ty'], pose['tz'], pose['rz']
                
                # 建立 3D 框頂點
                x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
                y_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
                z_corners = [0, 0, 0, 0, h, h, h, h]
                corners = np.vstack([x_corners, y_corners, z_corners])
                
                # 旋轉與平移
                c, s = np.cos(rz), np.sin(rz)
                R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
                corners = R @ corners
                corners[0, :] += tx
                corners[1, :] += ty
                corners[2, :] += tz
                
                # 投影至 2D 影像
                ones = np.ones((1, corners.shape[1]))
                corners_homo = np.vstack((corners, ones))
                img_coords = projection_matrix @ corners_homo
                x = img_coords[0, :] / img_coords[2, :]
                y = img_coords[1, :] / img_coords[2, :]
                pts_2d = np.vstack((x, y)).T.astype(int)
                
                # 繪製 12 條 3D 邊線
                lines = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]
                for start, end in lines:
                    cv2.line(img, tuple(pts_2d[start]), tuple(pts_2d[end]), (0, 255, 0), 2)
                    
        cv2.imshow("KITTI Tracklets Python Viewer", img)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
            
    cv2.destroyAllWindows()

# 執行方式 (需替換為你的 KITTI 資料集路徑)
run_visualization("/home/user/LiDAR_Camera_Perception_ws/data/2011_09_26/2011_09_26_drive_0009_sync")