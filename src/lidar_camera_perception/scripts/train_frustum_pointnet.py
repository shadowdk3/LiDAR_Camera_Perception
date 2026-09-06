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

import frustum_utils
    
if __name__ == "__main__":
    VISUALIZE_DATASET = False
    
    # 1. Automatically select GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=> Using device: {device}")
    
    data_path = "/home/user/LiDAR_Camera_Perception_ws/data/2011_09_26/2011_09_26_drive_0009_sync"
    dataset = frustum_utils.KittiFrustumDataset(data_path, "/home/user/LiDAR_Camera_Perception_ws/models/yolo11n.pt")
    log_path = "runs/frustum_pointnet_experiment_3d_corner_loss"
    
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
    
    # Configure Cosine Annealing Learning Rate Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
    
    l1_loss_fn = nn.SmoothL1Loss()
    corner_loss_fn = frustum_utils.Custom3DCornerLoss().to(device)

    # Training Loop across multiple epochs
    print(f"=> Starting training for {num_epochs} epochs across {len(train_loader)} training samples...")
    for epoch in range(num_epochs):
        # train
        model.train()
        total_train_loss = 0.0
        
        for batch_points, batch_gt_boxes in train_loader:
            # 3. Move batch data tensors to GPU
            batch_points = batch_points.to(device, non_blocking=True)
            batch_gt_boxes = batch_gt_boxes.to(device, non_blocking=True)
            
            optimizer.zero_grad()                                   # Clear previous gradients
            # Mixed precision forward pass
            with torch.amp.autocast('cuda'):
                predictions = model(batch_points)
                
                # Compute the custom 3D corner loss
                loss_l1 = l1_loss_fn(predictions, batch_gt_boxes)
                loss_corner = corner_loss_fn(predictions, batch_gt_boxes)   
            
                # Combine them with a weight (e.g., 80% L1, 20% Corner Loss)
                loss = loss_l1 + (0.2 * loss_corner)

            # Scaled backward pass
            scaler.scale(loss).backward()                          # Backward pass (compute gradients)
            scaler.step(optimizer)                                 # Update model weights
            scaler.update()
            total_train_loss += loss.item()
            
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Eval
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch_points, batch_gt_boxes in val_loader:
                batch_points = batch_points.to(device, non_blocking=True)
                batch_gt_boxes = batch_gt_boxes.to(device, non_blocking=True)
                
                with torch.amp.autocast('cuda'):
                    predictions = model(batch_points)
                    # val_loss = F.smooth_l1_loss(predictions, batch_gt_boxes)
                    val_loss = corner_loss_fn(predictions, batch_gt_boxes)    
                    
                total_val_loss += val_loss.item()
                
        avg_val_loss = total_val_loss / len(val_loader)
        
        # Step the learning rate scheduler
        scheduler.step()
        
        print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")        
        
        # Log training loss to TensorBoard
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Loss/Val', avg_val_loss, epoch)
        writer.add_scalar('LearningRate', scheduler.get_last_lr()[0], epoch)
        
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
            torch.save(checkpoint, 'frustum_pointnet_checkpoint.pth')
            print("=> Saved best training checkpoint.")