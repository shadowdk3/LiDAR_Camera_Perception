# LiDAR-Camera Perception Pipeline & Database System
----------------------------------

## Prerequisites & Installation

* Install PostgreSQL and the PostGIS spatial extensions on your Ubuntu/ROS 2 environment:

```
sudo apt install -y postgresql postgresql-contrib postgis postgresql-16-postgis-3
```
*(Note: Change `-16-` to your corresponding PostgreSQL version if needed, e.g., 14 or 15).*

* Ensure your ROS 2 workspace workspace or virtual environment has the required libraries:

```
pip install psycopg2-binary
```

## Database & Schema Initialization

Follow these steps to initialize the relational database and create the optimized tables for your perception outputs.

* Log in to PostgreSQL CLI

```
sudo -i -u postgres psql
```

* Configure Password & Create Database

Run the following SQL commands to secure your administration account and spin up a dedicated database:

```
CREATE DATABASE lidar_perception;
ALTER USER postgres WITH PASSWORD '123456';
```

* Switch Context and Enable PostGIS

Connect to your new database, and enable the PostGIS spatial geometry extension (Crucial for BBox storage)

```
\c lidar_perception
CREATE EXTENSION postgis;
```

* Create the Detections Table & Indexing

Execute this block to build the table structure designed for high-frequency 10 FPS tracking data:

```
CREATE TABLE camera_yolo_detections (
    id SERIAL PRIMARY KEY,
    sec INT NOT NULL,
    nanosec INT NOT NULL,
    frame_id VARCHAR(100) NOT NULL,
    class_name VARCHAR(50) NOT NULL,
    confidence REAL NOT NULL,
    center_x REAL NOT NULL,
    center_y REAL NOT NULL,
    width REAL NOT NULL,
    height REAL NOT NULL,
    bbox_2d_polygon geometry(Polygon, 0)
);
```

Generate spatial-temporal composite indexes for fast real-time 10 FPS lookups

```
CREATE INDEX idx_yolo_sec_nanosec ON camera_yolo_detections(sec, nanosec);
```

* Verify Your Tables

```
\dt
```

**Expected Output:**

```
                 List of relations
 Schema |          Name          | Type  |  Owner   
--------+------------------------+-------+----------
 public | camera_yolo_detections | table | postgres
 public | spatial_ref_sys        | table | postgres
(2 rows)
```

* Exit the interface:

```
\q
```

## Quick Start & Testing

### Test Database Ingestion (Standalone Script)
Run the isolated test script to verify database connectivity, credential mapping, and geometry string parsing without launching ROS 2:

```
python3 test_db.py
```

## Clear data in table

```
TRUNCATE TABLE camera_yolo_detections RESTART IDENTITY;
```