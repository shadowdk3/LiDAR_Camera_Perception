# LiDAR Tracker
------------------------------------

- Use `Kalman Filter` track object

## Kalman Filter

```
       ┌─────────────────────────────────────────────────────────┐
       │                                                         │
       │                 1. Predict Phase                        │
       │      "Estimate where the object should be now           │
       │       based on its last known speed and direction"      │
       │                                                         │
       │       • State Predict:       X = F * X                  │
       │       • Uncertainty (P) increases  (purely guessing)    │
       │                                                         │
       └────────────────────────────┬────────────────────────────┘
                                    │
                                    │ Outputs: Predicted Position
                                    ▼
       ┌─────────────────────────────────────────────────────────┐
       │                                                         │
       │                 2. Kalman Gain (K)                      │
       │      "The Scale of Trust: Decide who to trust more:     │
       │       the physical prediction OR the sensor hardware?"  │
       │                                                         │
       │       • If sensor noise is HIGH ──► Trust Prediction    │
       │       • If sensor noise is LOW  ──► Trust Sensor        │
       │                                                         │
       └────────────────────────────┬────────────────────────────┘
                                    │
     + New Sensor Data (LiDAR)      │ Outputs: Trust Weight (K)
     "Real-world observation        ▼
      loaded with sensor noise"┌─────────────────────────────────────────────────────────┐
       │                       │                                                         │
       └──────────────────────►│                 3. Update Phase                         │
                               │      "Fuse both worlds to calculate the absolute        │
                               │       best mathematical guess of the truth"             │
                               │                                                         │
                               │       • State Correct: X = Predict + K * (Sensor - Pred)│
                               │       • Uncertainty (P) shrinks (highly confident now)  │
                               │                                                         │
                               └────────────────────────────┬────────────────────────────┘
                                                            │
                                                            └─────── (Loop back to Step 1 for next frame)
```

- X = F * X

```
    [ Predicted State ]     [  State Transition Matrix (F)  ]     [ Current State ]
    [      (6 x 1)    ]     [            (6 x 6)            ]     [     (6 x 1)   ]

       ┌            ┐       ┌                               ┐     ┌            ┐
       │   x_pred   │       │  1   0   0   dt   0   0       │     │     x      │
       │   y_pred   │       │  0   1   0   0    dt  0       │     │     y      │
       │   z_pred   │   =   │  0   0   1   0    0   dt      │  *  │     z      │
       │  vx_pred   │       │  0   0   0   1    0   0       │     │     vx     │
       │  vy_pred   │       │  0   0   0   0    1   0       │     │     vy     │
       │  vz_pred   │       │  0   0   0   0    0   1       │     │     vz     │
       └            ┘       └                               ┘     └            ┘
```

- X: State Vector, a matrix holds the physical attribute of the object, is a 6*1 column vector storing position and velocity:
        ┌   ┐
        │ x │
        │ y │  (3D Position: where the object is)
   X =  │ z │
        │ Vx│  
        │ Vy│  (3D Velocity: how fast it is moving)
        │ Vz|
        └   ┘
  Right-side X: The verified state from the previous frame
  Left-side X: The predicted state for the upcoming frame

- F: State Transition Matrix, advances your state vector forward by a small slice of time (dt)

- 3D Position Shift :  x_pred  =  x  +  vx * dt
                       y_pred  =  y  +  vy * dt
                       z_pred  =  z  +  vz * dt

- 3D Velocity Coast :  vx_pred =  vx
                       vy_pred =  vy
                       vz_pred =  vz
                        
- The Kalman Filter Loop in 3 Sentences

1. Predict: The system uses a physics motion model to project the object's next position forward in time, which inherently increases estimation uncertainty.

2. Weigh: When new sensor data arrives, the system calculates the Kalman Gain (K) to determine whether to trust the theoretical physics prediction or the noisy hardware measurements more.

3. Update: It mathematically fuses both inputs to calculate the optimal, highly confident "true" state, drastically shrinking uncertainty before restarting the cycle.

## ROS2

```
       LiDAR Bounding Boxes
                 │
                 ▼
       ┌──────────────────┐
       │    New Frame     │
       │ Object Detection │
       └────────┬─────────┘
                │
                ▼
           Predict Phase
                │
                │ Kalman Filter
                │
                ▼
       Next Track Position
                │
                ▼
         Data Association
       "Which detection 
        belongs to which track?"
                │
                ▼
           Update Phase
       Correct Kalman State
       with Sensor Detection
                │
                ▼
         Track Management
       New object  → Create ID
       Lost too long → Erase ID
                │
                ▼
      /lidar/tracked_objects
```

- Receives input 3D bounding boxes from the LiDAR cluster object `/lidar_cluster/bounding_boxes`
- Coasts existing tracks forward using physics matrices (X = F × X) to guess their positions in the new frame.
- Pairs the predicted track locations with the incoming detections using distance metrics and optimization algorithms
- Fuses the paired sensor data into the Kalman Filter to correct the track's speed and position while shrinking uncertainty.
- Spawns new track IDs for unmatched detections, and permanently purges dead tracks using the erase/remove_if idiom
- Publishes the clean, smoothed, and ID-consistent list of targets to the `/lidar/tracked_objects`

![lider_tracker](./reference/lider_tracker.gif)


