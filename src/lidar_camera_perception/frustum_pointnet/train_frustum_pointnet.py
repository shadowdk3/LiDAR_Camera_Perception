"""
Define Dataset (KITTI path parsing, YOLO frustum extraction, padding to 512 pts)
Create DataLoader (batch_size=4, shuffle=True)
Initialize model & optimizer (SimpleFrustumPointNet, Adam lr=0.001)
Training loop: forward pass, loss computation, zero_grad, backward, and step
Save best model weights (lowest loss) to 'frustum_pointnet.pth'
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D
                    
from utils import frustum_utils

if __name__ == "__main__":
    VISUALIZE_DATASET = False
    
    # 1. Automatically select GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=> Using device: {device}")
    
    data_path = "/home/user/LiDAR_Camera_Perception_ws/data/2011_09_26/2011_09_26_drive_0009_sync"
    log_path = "runs/frustum_pointnet_experiment"
    model_path = "checkpoints/frustum_pointnet_checkpoint.pth"
    
    dataset = frustum_utils.KittiFrustumDataset(data_path, "/home/user/LiDAR_Camera_Perception_ws/models/yolo11n.pt")
    
    if VISUALIZE_DATASET:
        print(f"=> Starting dataset visualization validation. Total samples: {len(dataset)}")

        for idx in range(len(dataset)):
            # Retrieve tensor data
            sampled_pts, gt_box = dataset[idx]
            
            # Retrieve original paths and data from cache for visualization
            raw_sample = dataset.cached_data[idx]
            
            print(f"Viewing sample {idx + 1} / {len(dataset)}...")
            
            # Call the Open3D visualization function
            # visualize_sample reads raw_sample['bin_path'] and raw_sample['img_path'], then draws the 3D bounding box using gt_box
            frustum_utils.visualize_sample(
                bin_path=raw_sample['bin_path'], 
                img_path=raw_sample['img_path'], 
                gt_box=raw_sample['gt_box'] # Recommended to use either the raw gt_box or the centroid-aligned gt_box for inspection
            )
            
    # Split dataset into train (80%) and validation (20%) sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0, pin_memory=True)
    
    # 2. Move model to GPU
    model = frustum_utils.SimpleFrustumPointNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=log_path)
    
    # Enable Automatic Mixed Precision (AMP) for faster FP16 training and lower memory overhead
    scaler = torch.amp.GradScaler('cuda')

    best_loss = float('inf')
    num_epochs = 100
    
    learning_rate = 1e-5
    
    # Configure Cosine Annealing Learning Rate Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=learning_rate)
    
    corner_loss_fn = frustum_utils.Custom3DCornerLoss().to(device)

    # Training Loop across multiple epochs
    print(f"=> Starting training for {num_epochs} epochs across {len(train_loader)} training samples...")
    for epoch in range(num_epochs):
        # train
        model.train()
        total_train_loss = 0.0
        total_train_center_loss = 0.0
        total_train_size_loss = 0.0
        total_train_corner_loss = 0.0
        
        for batch_points, batch_gt_boxes in train_loader:
            # 3. Move batch data tensors to GPU
            batch_points = batch_points.to(device, non_blocking=True)
            batch_gt_boxes = batch_gt_boxes.to(device, non_blocking=True)
            
            optimizer.zero_grad()                                   # Clear previous gradients
            # Mixed precision forward pass
            with torch.amp.autocast('cuda'):
                predictions = model(batch_points)
                
                # 1. Center loss (x, y, z)
                loss_center = nn.SmoothL1Loss()(predictions[:, :3], batch_gt_boxes[:, :3])
                
                # 2. Size loss (l, w, h) - Add this to prevent shrinking
                loss_size = nn.SmoothL1Loss()(predictions[:, 3:6], batch_gt_boxes[:, 3:6])

                # 3. Corner loss for overall geometry and rotation
                loss_corner = corner_loss_fn(predictions, batch_gt_boxes)
                
                # Combined Loss (weight size slightly higher to expand boxes)
                loss = (2.0 * loss_center) + (1 * loss_size) + (1 * loss_corner)
                
            # Scaled backward pass
            scaler.scale(loss).backward()                          # Backward pass (compute gradients)
            scaler.step(optimizer)                                 # Update model weights
            scaler.update()
            
            # acc batch loss
            total_train_loss += loss.item()
            total_train_center_loss += loss_center.item()
            total_train_size_loss += loss_size.item()
            total_train_corner_loss += loss_corner.item()
            
        # avg epoch loss
        num_train_batches = len(train_loader)
        avg_train_loss = total_train_loss / num_train_batches
        avg_train_center = total_train_center_loss / num_train_batches
        avg_train_size = total_train_size_loss / num_train_batches
        avg_train_corner = total_train_corner_loss / num_train_batches
        
        # Eval
        model.eval()
        total_val_loss = 0.0
        total_val_center = 0.0
        total_val_size = 0.0
        total_val_corner = 0.0
        
        # Prepare the container for plotting TensorBoard BEV (we only take the first batch from val_loader for plotting)
        sample_bev_fig = None
        
        with torch.no_grad():
            for batch_idx, (batch_points, batch_gt_boxes) in enumerate(val_loader):
                batch_points = batch_points.to(device, non_blocking=True)
                batch_gt_boxes = batch_gt_boxes.to(device, non_blocking=True)
                
                with torch.amp.autocast('cuda'):
                    predictions = model(batch_points)
                    val_loss_center = nn.SmoothL1Loss()(predictions[:, :3], batch_gt_boxes[:, :3]) 
                    val_loss_size = nn.SmoothL1Loss()(predictions[:, 3:6], batch_gt_boxes[:, 3:6])
                    val_loss_corner = corner_loss_fn(predictions, batch_gt_boxes)
                    
                    val_loss = (2 * val_loss_center) + (1 * val_loss_size) + (1 * val_loss_corner)
                    
                total_val_loss += val_loss.item()
                total_val_center += val_loss_center.item()
                total_val_size += val_loss_size.item()
                total_val_corner += val_loss_corner.item()
                
                # Allows you to see the BEV visualization in TensorBoard at each epoch before training finishes
                if batch_idx % 10 == 0:
                    fig, ax = plt.subplots(figsize=(6, 6))
                    
                    # Get the first item of data from this batch for plotting
                    pts_np = batch_points[0].cpu().numpy() # (512, 3)
                    pred_np = predictions[0].cpu().numpy() # (7,)
                    gt_np = batch_gt_boxes[0].cpu().numpy() # (7,)
                    
                    # Plot normalized point cloud (x, y)
                    ax.scatter(pts_np[:, 0], pts_np[:, 1], s=1, c='gray', label='Normalized Points')
                    # Plot predicted box center (red cross)
                    ax.scatter(pred_np[0], pred_np[1], c='red', marker='x', s=100, label='Pred Center')
                    # Plot ground truth box center (green circle)
                    ax.scatter(gt_np[0], gt_np[1], c='green', marker='o', s=80, label='GT Center')
                    
                    # Create and plot the 2D bounding box with width, length, and yaw angle (BEV Bounding Box)
                    def draw_bev_box(box_params, color, label_text):
                        x, y, z, l, w, h, rz = box_params
                        # Matplotlib's Rectangle uses the bottom-left corner as its reference point, 
                        # so we need to subtract half the length and width to shift it back to center.
                        # Note: In KITTI format, l usually corresponds to car length (forward direction) and w to car width.
                        rect = Rectangle((-w/2, -l/2), w, l, linewidth=2, edgecolor=color, facecolor='none', label=label_text)
                        
                        # Create rotation and translation transformations
                        transform = Affine2D().rotate_deg_around(0, 0, np.degrees(rz)).translate(x, y) + ax.transData
                        rect.set_transform(transform)
                        ax.add_patch(rect)
                        
                    draw_bev_box(pred_np, color='red', label_text='Pred Box')
                    draw_bev_box(gt_np, color='green', label_text='GT Box')
                    
                    ax.set_title(f"Epoch {epoch+1} BEV Center Check")
                    ax.legend(loc='upper right')
                    ax.grid(True)
                    
                    sample_bev_fig = fig
                    plt.close(fig) # Add this line to prevent memory leaks and warnings
                    
        num_val_batches = len(val_loader)
        avg_val_loss = total_val_loss / num_val_batches
        avg_val_center = total_val_center / num_val_batches
        avg_val_size = total_val_size / num_val_batches
        avg_val_corner = total_val_corner / num_val_batches
        
        # Step the learning rate scheduler
        scheduler.step()
        
        print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")        
        
        # Write the matplotlib figure to TensorBoard
        if sample_bev_fig is not None:
            writer.add_figure('Visual/Val_BEV', sample_bev_fig, epoch)
            plt.close(sample_bev_fig)
            
        # Log training loss to TensorBoard
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Loss/Val', avg_val_loss, epoch)
        writer.add_scalar('LearningRate', scheduler.get_last_lr()[0], epoch)
        
        # Log individual loss components
        writer.add_scalar('Loss/Train_Center', avg_train_center, epoch)
        writer.add_scalar('Loss/Train_Size', avg_train_size, epoch)
        writer.add_scalar('Loss/Train_Corner', avg_train_corner, epoch)
        
        writer.add_scalar('Loss/Val_Center', avg_val_center, epoch)
        writer.add_scalar('Loss/Val_Size', avg_val_size, epoch)
        writer.add_scalar('Loss/Val_Corner', avg_val_corner, epoch)
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            # Save comprehensive checkpoint dictionary
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_loss': best_loss
            }
            torch.save(checkpoint, model_path)
            print("=> Saved best training checkpoint.")