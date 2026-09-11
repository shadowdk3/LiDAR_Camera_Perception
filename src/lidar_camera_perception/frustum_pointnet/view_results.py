import argparse
import torch
import numpy as np
import onnxruntime as ort

from utils import frustum_utils

def view_model_results_with_sample(data_path, yolo_path, model_path, multi_box):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = frustum_utils.KittiFrustumDataset(data_path, yolo_path)
    
    is_onnx = model_path.endswith('.onnx')
    
    if is_onnx:
        print(f"=> Loading ONNX model from {model_path}...")
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
        ort_session = ort.InferenceSession(model_path, providers=providers)
        input_name = ort_session.get_inputs()[0].name
    else:
        print(f"=> Loading PyTorch model from {model_path}...")
        model = frustum_utils.FrustumPointNetV2().to(device)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
    
    print(f"=> Starting result visualization. Total samples: {len(dataset)}")

    if multi_box:
        # Group dataset indices by frame (bin_path) to collect multiple objects per scan
        frame_dict = {}
        for idx in range(len(dataset)):
            raw_sample = dataset.cached_data[idx]
            bin_path = raw_sample['bin_path']
            if bin_path not in frame_dict:
                frame_dict[bin_path] = []
            frame_dict[bin_path].append(idx)

        print(f"=> Grouped into {len(frame_dict)} unique frames for multi-box viewing.")
        
        with torch.no_grad():
            for frame_idx, (bin_path, indices) in enumerate(frame_dict.items()):
                gt_boxes_list = []
                pred_boxes_list = []
                img_path = None

                for idx in indices:
                    sampled_pts, gt_box_tensor, batch_mark = dataset[idx]
                    raw_sample = dataset.cached_data[idx]
                    img_path = raw_sample['img_path']
                    
                    frustum_pts = raw_sample['frustum_pts']
                    centroid = np.mean(frustum_pts, axis=0)
                
                    if is_onnx:
                        batch_points = sampled_pts.unsqueeze(0).numpy().astype(np.float32)
                        outputs = ort_session.run(None, {input_name: batch_points})
                        pred_box = outputs[0][0]
                    else:
                        batch_points = sampled_pts.unsqueeze(0).to(device)
                        predictions, logits, center_offset = model(batch_points)
                        pred_box = predictions[0].cpu().numpy()
                
                    pred_box[0] += centroid[0]
                    pred_box[1] += centroid[1]
                    pred_box[2] += centroid[2]
                    
                    gt_boxes_list.append(raw_sample['gt_box'])
                    pred_boxes_list.append(pred_box)

                print(f"\nViewing frame {frame_idx + 1} / {len(frame_dict)} (Objects: {len(indices)})...")
                
                frustum_utils.visualize_all_sample(
                    bin_path=bin_path,
                    gt_boxes=np.array(gt_boxes_list),
                    pred_boxes=np.array(pred_boxes_list)
                )
    else:
        with torch.no_grad():
            for idx in range(len(dataset)):
                sampled_pts, gt_box_tensor, batch_mark = dataset[idx]
                raw_sample = dataset.cached_data[idx]
                
                # Re-compute the exact same centroid used during dataset normalization
                frustum_pts = raw_sample['frustum_pts']
                centroid = np.mean(frustum_pts, axis=0)
            
                # Run model inference
                if is_onnx:
                    # Shape: [1, 512, 3]
                    batch_points = sampled_pts.unsqueeze(0).numpy().astype(np.float32)
                    outputs = ort_session.run(None, {input_name: batch_points})
                    pred_box = outputs[0][0] # Assuming first output contains bounding box predictions
                else:
                    batch_points = sampled_pts.unsqueeze(0).to(device)
                    predictions, logits, center_offset = model(batch_points)
                    pred_box = predictions[0].cpu().numpy()
                
                # Map the prediction back to absolute camera coordinates by adding the centroid
                pred_box[0] += centroid[0]
                pred_box[1] += centroid[1]
                pred_box[2] += centroid[2]
                
                print(f"\nViewing sample {idx + 1} / {len(dataset)}...")
                
                # Call visualization function with both GT and Prediction
                frustum_utils.visualize_sample(
                    bin_path=raw_sample['bin_path'],
                    gt_box=raw_sample['gt_box'],
                    pred_box=pred_box
                )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View Frustum PointNet results using PyTorch or ONNX.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to .pth or .onnx model file")
    parser.add_argument("--data_path", type=str, default="/home/user/LiDAR_Camera_Perception_ws/data/2011_09_26/2011_09_26_drive_0009_sync")
    parser.add_argument("--yolo_path", type=str, default="/home/user/LiDAR_Camera_Perception_ws/models/yolo11n.pt")
    parser.add_argument("--multi_box", action="store_true", default=False, help="View multiple bounding boxes in one image window")
    
    args = parser.parse_args()
    
    view_model_results_with_sample(args.data_path, args.yolo_path, args.model_path, args.multi_box)