#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl_conversions/pcl_conversions.h>

class PreprocessingNode : public rclcpp::Node
{
    public:
        PreprocessingNode() : Node("lidar_preprocessing_node")
        {
            // Create a subscription to the LIDAR point cloud topic
            subscription_lidar = this->create_subscription<sensor_msgs::msg::PointCloud2>(
                "/kitti/point_cloud",
                10,
                std::bind(&PreprocessingNode::lidarCallback, this, std::placeholders::_1)
            );
            
            // Create publishers for ground and non-ground point clouds
            publisher_object_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/lidar_preprocessing/object_pcd", 10);
            publisher_ground_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/lidar_preprocessing/ground_pcd", 10);

            RCLCPP_INFO(this->get_logger(), "Lidar Preprocessing Node Started...");
        }

    private:
        // Shortcut alias for point cloud pointers to keep the code clean
        using PointCloudPtr = pcl::PointCloud<pcl::PointXYZ>::Ptr;

        float ground_threshold = -1.5;  // Threshold to separate ground and non-ground points
        
        /**
        * Voxel down-samples the input point cloud.
        * @param input_cloud The original input point cloud.
        * @return A new point cloud with reduced density.
        */
        PointCloudPtr voxelDownsample(const PointCloudPtr& input_cloud)
        {
            PointCloudPtr downsampled_cloud(new pcl::PointCloud<pcl::PointXYZ>());
            
            pcl::VoxelGrid<pcl::PointXYZ> voxel;
            voxel.setInputCloud(input_cloud);
            voxel.setLeafSize(0.1f, 0.1f, 0.1f);        //  Set the voxel grid leaf size (0.1m x 0.1m x 0.1m)
            voxel.filter(*downsampled_cloud);

            return downsampled_cloud;
        }

        /**
        * Crops the point cloud to a specific 3D Region of Interest (ROI).
        * @param input_cloud The original input point cloud.
        * @return A new point cloud containing only the points inside the 3D box.
        */
        PointCloudPtr cropROI(const PointCloudPtr& input_cloud)
        {
            PointCloudPtr cropped_cloud(new pcl::PointCloud<pcl::PointXYZ>());
            
            pcl::CropBox<pcl::PointXYZ> crop;
            crop.setInputCloud(input_cloud);
            
            // Minimum Boundary: (-20.0, -10.0, -3.0)  // (backward, right, down)
            // X = -20.0: Limits the view up to 20 meters behind the sensor.
            // Y = -10.0: Limits the view up to 10 meters to the right of the sensor.
            // Z = -3.0: Limits the view up to 3 meters below the sensor's origin.
            crop.setMin(Eigen::Vector4f(-20.0, -10.0, -3.0, 1.0));  // define the Minimum Boundary

            // Maximum Boundary: (50.0, 10.0, 5.0)     // (forward, left, up)
            // X = 50.0: Limits the view up to 50 meters ahead of the sensor.
            // Y = 10.0: Limits the view up to 10 meters to the left of the sensor.
            // Z = 5.0: Limits the view up to 5 meters above the sensor.
            crop.setMax(Eigen::Vector4f(50.0, 10.0, 5.0, 1.0));     // define the Maximum Boundary

            // Execute filtering and save results to cropped_cloud
            crop.filter(*cropped_cloud);
            return cropped_cloud;
        }

        /**
        * Checks if ground points are detected in the ROI point cloud.
        * @param roi_cloud The cropped input point cloud.
        * @param bottom_cloud Output container for points below the height threshold.
        * @param top_cloud Output container for points at or above the height threshold.
        * @return True if ground points are detected, false otherwise.
        */
        bool checkGroundDetected(const PointCloudPtr& roi_cloud, 
                                PointCloudPtr& bottom_cloud, 
                                PointCloudPtr& top_cloud)
        {
            // Splits ROI point cloud into two separate groups based on height (Z-axis)
            // Aim to minimize the number of points in the bottom cloud, which is expected to contain ground points.
           
            // Clear containers to prevent accumulation over consecutive callback frames
            bottom_cloud->points.clear();
            top_cloud->points.clear();

            // Loop through and split points based on the class ground_threshold member variable
            for (const auto &point : roi_cloud->points)
            {
                if (point.z < ground_threshold)                 
                {
                    bottom_cloud->points.push_back(point);
                }
                else
                {
                    top_cloud->points.push_back(point);
                }
            }

            // if no points in the bottom cloud, return early to avoid unnecessary processing
            if (bottom_cloud->empty())
                return false;

            // Properly update PCL metadata for the output clouds
            bottom_cloud->width = bottom_cloud->points.size();
            bottom_cloud->height = 1;
            bottom_cloud->is_dense = true;

            top_cloud->width = top_cloud->points.size();
            top_cloud->height = 1;
            top_cloud->is_dense = true;

            return true;
        }

        /**
         * Separates the point cloud into ground (bottom) and obstacles (top).
         * @param roi_cloud The cropped input point cloud.
         * @param ground_cloud Output container for ground points.
         * @param object_cloud Output container for obstacle points.
         * @return True if separation is successful, false otherwise.
         */
        bool separateGroundAndObstacles(const PointCloudPtr& roi_cloud, 
                                        PointCloudPtr& ground_cloud,
                                        PointCloudPtr& object_cloud)
        {
            // Clear containers to prevent accumulation over consecutive callback frames
            ground_cloud->points.clear();
            object_cloud->points.clear();

            // RANSAC Plane Segmentation
            pcl::SACSegmentation<pcl::PointXYZ> seg;
            pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
            pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());

            seg.setOptimizeCoefficients(true);
            seg.setModelType(pcl::SACMODEL_PLANE);
            seg.setMethodType(pcl::SAC_RANSAC);
            seg.setDistanceThreshold(0.25);
            seg.setMaxIterations(1000);
            seg.setInputCloud(roi_cloud);
            seg.segment(*inliers, *coefficients);

            if (inliers->indices.empty())
                return false;

            // Extract Ground & object
            pcl::ExtractIndices<pcl::PointXYZ> extract;

            extract.setInputCloud(roi_cloud);
            extract.setIndices(inliers);

            // Ground
            extract.setNegative(false);
            extract.filter(*ground_cloud);

            // object (bottom part)
            extract.setNegative(true);
            extract.filter(*object_cloud);

            // Properly update PCL metadata for the output clouds
            ground_cloud->width = ground_cloud->points.size();
            ground_cloud->height = 1;
            ground_cloud->is_dense = true;

            object_cloud->width = object_cloud->points.size();
            object_cloud->height = 1;
            object_cloud->is_dense = true;

            return true;
        }

        void lidarCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
        {
            // Convert ROS2 PointCloud2 to PCL
            PointCloudPtr cloud(new pcl::PointCloud<pcl::PointXYZ>());
            pcl::fromROSMsg(*msg, *cloud);

            if (cloud->empty())
                return;

            // Voxel Downsampling
            PointCloudPtr cloud_filtered = voxelDownsample(cloud);

            // ROI Filtering
            PointCloudPtr cloud_roi = cropROI(cloud_filtered);

            // Separate Ground and Obstacles
            PointCloudPtr bottom_cloud(new pcl::PointCloud<pcl::PointXYZ>());
            PointCloudPtr top_cloud(new pcl::PointCloud<pcl::PointXYZ>());

            // Check if ground points are detected; if not, return early
            if (!checkGroundDetected(cloud_roi, bottom_cloud, top_cloud))
                return;

            // Extract Ground & bottom non-ground points using RANSAC Plane Segmentation
            PointCloudPtr ground_cloud(new pcl::PointCloud<pcl::PointXYZ>());
            PointCloudPtr extract_bottom_object_cloud(new pcl::PointCloud<pcl::PointXYZ>());

            // If ground points are not detected, return early to avoid publishing empty clouds
            if (!separateGroundAndObstacles(bottom_cloud, ground_cloud, extract_bottom_object_cloud))
                return;

            // Combine top + bottom object clouds to form the final object cloud
            PointCloudPtr object_cloud(new pcl::PointCloud<pcl::PointXYZ>());
            *object_cloud = *top_cloud + *extract_bottom_object_cloud;

            // Publish the ground and object point clouds
            sensor_msgs::msg::PointCloud2 ground_msg;
            pcl::toROSMsg(*ground_cloud, ground_msg);
            ground_msg.header = msg->header;
            publisher_ground_->publish(ground_msg);

            sensor_msgs::msg::PointCloud2 object_msg;
            pcl::toROSMsg(*object_cloud, object_msg);
            object_msg.header = msg->header;
            publisher_object_->publish(object_msg);
        }

        // ROS2 Subscribers and Publishers
        rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_lidar;
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_object_;
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_ground_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);                               // Initialize ROS 2
    rclcpp::spin(std::make_shared<PreprocessingNode>());    // Spin the node to process callbacks
    rclcpp::shutdown();
    return 0;
}