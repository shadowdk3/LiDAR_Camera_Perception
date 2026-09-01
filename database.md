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

## Metabase Data Visualization 

Demonstrates how to use Metabase (an open-source business intelligence and data visualization platform) to connect to a local PostgreSQL database and transform AI detection metadata (such as camera_yolo_detections) into interactive bar and pie charts without writing SQL queries.

### Configure PostgreSQL for External/Docker Access

1. To allow Metabase running inside a Docker container to access your host machine's PostgreSQL database, you need to update your database network configurations.

```
sudo pkill -u postgres
sudo service postgresql start
```

2. Modify the Core Configuration File (Listen on all IP addresses)

```
sudo nano /etc/postgresql/16/main/postgresql.conf
```

Find the line # listen_addresses = '*' and remove the # comment symbol to uncomment it, then save the file.

3. Modify the Client Authentication Configuration File

```
sudo nano /etc/postgresql/16/main/pg_hba.conf
```

Add the following line to the end of the file to allow incoming Docker connections, then save:

```
host    all             all             0.0.0.0/0               scram-sha-256
```

### Launch Metabase via Docker

1. Open your Terminal or Command Prompt and execute the following command to download and run Metabase in the background:

```
docker run -d -p 3000:3000 --name metabase metabase/metabase
```

This maps the Metabase web interface to host port 3000.

### Connect Metabase to PostgreSQL

1. Open your web browser and navigate to the Metabase setup page at http://localhost:3000

2. Fill in the following database connection details during the initialization wizard:

| Field Name | Value / Description|
| Host | Enter `host.docker.internal` (Resolves to the host/WSL machine on Windows/Mac Docker).If connection fails, try using 172.17.0.1.|
| Port| 5432 | 
| Database name | Your database name |
| Username | Your database username |
| Password | Your database password |

3. Click Save. Once authentication succeeds, you will see your camera_yolo_detections tables listed in the dashboard.

### Create AI Label Distribution Charts

Once connected, you can use Metabase’s no-code graphical interface to build visual insights and quickly track which AI labels (e.g., person, car) occur most frequently:

1. Aggregate and Group Data

* Select your `camera_yolo_detections` data table.
* Click through the query settings menu: `Summarize` -> `Summarize by` -> `Group by` -> `Class Name`.

![group_data](./reference/group_data.gif)
![group_data2](./reference/group_data2.gif)

2. Select a Visualization Type

* Go to the visualization selector in the bottom left corner and pick either `Pie` or `Bar` chart.

![group_data_pie](./reference/group_data_pie.gif)

3. Configure Numeric Binning (If Applicable)

* When handling coordinates, confidence values, or custom numerical metrics, you can open the binning menu to group ranges using:
    * Auto binned (Automatic division)
    * 10 bins (Divide data into 10 intervals)
    * 50 bins (Divide data into 50 intervals)
    * Don't bin (Keep individual raw values)

![bin_8](./reference/bin_8.gif)
![bin_10](./reference/bin_10.gif)
