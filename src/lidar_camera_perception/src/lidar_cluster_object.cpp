#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/search/kdtree.h>
#include <pcl/common/common.h>

class LidarClusterObjectNode : public rclcpp::Node
{
    public:
        LidarClusterObjectNode() : Node("lidar_cluster_detector_node")
        {
            // Create a subscription to the obstacle point cloud topic
            subscription_object_pcd_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
                "/lidar_preprocessing/object_pcd", 10,
                std::bind(&LidarClusterObjectNode::clusterObjectCallback, this, std::placeholders::_1));

            // Create publishers for clustered obstacles and bounding boxes
            publisher_obstacles_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
                "/lidar_cluster/clustered_obstacles_pcd", 10);

            publisher_bbox_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
                "/lidar_cluster/bounding_boxes", 10);

            RCLCPP_INFO(this->get_logger(), "LiDAR Cluster Object Node Started...");
        }

    private:
        // Shortcut alias for point cloud pointers to keep the code clean
        using PointCloudPtr = pcl::PointCloud<pcl::PointXYZ>::Ptr;

        /**
         * Separates ground and obstacles in the point cloud
         * @param roi_cloud The region of interest point cloud
         * @param cluster_indices The indices of the clusters
         * @return True if successful, false otherwise
         */
        bool separateGroundAndObstacles(const PointCloudPtr& roi_cloud, 
                                        std::vector<pcl::PointIndices>& cluster_indices)
        {
            cluster_indices.clear(); 

            // Create a KdTree for fast spatial searching
            pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
            tree->setInputCloud(roi_cloud);

            pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
            ec.setClusterTolerance(0.6);   // eps = 0.6m
            ec.setMinClusterSize(30);      
            ec.setMaxClusterSize(25000);
            ec.setSearchMethod(tree);
            ec.setInputCloud(roi_cloud);
            ec.extract(cluster_indices);

            if (cluster_indices.empty())
                return false;
            return true;
        }

        /**
         * Processes a single cluster and extracts valid obstacles and bounding boxes
         * @param global_cloud The global point cloud
         * @param indices The indices of the cluster
         * @param header The message header
         * @param obstacle_id The ID of the obstacle
         * @param valid_obstacles_out The output point cloud for valid obstacles
         * @param marker_out The output bounding box marker
         * @return True if successful, false otherwise
         */
        bool processSingleCluster(
            const pcl::PointCloud<pcl::PointXYZ>::Ptr& global_cloud,
            const pcl::PointIndices& indices,
            const std_msgs::msg::Header& header,
            int obstacle_id,
            pcl::PointCloud<pcl::PointXYZ>& valid_obstacles_out,
            visualization_msgs::msg::Marker& marker_out)
        {
            // Skip clusters that are too small
            if (indices.indices.size() < 30)
                return false;

            // Populate the cluster point cloud with points from the original cloud
            pcl::PointCloud<pcl::PointXYZ> cluster;
            for (int idx : indices.indices)
                cluster.points.push_back(global_cloud->points[idx]);

            // Compute AABB (Axis-Aligned Bounding Box)
            pcl::PointXYZ min_pt, max_pt;
            pcl::getMinMax3D(cluster, min_pt, max_pt);

            float size_x = max_pt.x - min_pt.x;
            float size_y = max_pt.y - min_pt.y;
            float size_z = max_pt.z - min_pt.z;

            float max_horizontal = std::max(size_x, size_y);
            float min_horizontal = std::min(size_x, size_y);

            // Geometric Filtering 
            if (max_horizontal > 6.0 || min_horizontal < 1.1) return false;
            if (size_z > 3.5 || size_z < 0.8) return false;

            // combine cluster object into one pcd
            valid_obstacles_out += cluster;

            // Create Bounding Box Marker
            marker_out.header = header;
            marker_out.ns = "detected_obstacle";
            marker_out.id = obstacle_id;
            marker_out.type = visualization_msgs::msg::Marker::CUBE;
            marker_out.action = visualization_msgs::msg::Marker::ADD;

            marker_out.pose.position.x = (min_pt.x + max_pt.x) / 2.0;
            marker_out.pose.position.y = (min_pt.y + max_pt.y) / 2.0;
            marker_out.pose.position.z = (min_pt.z + max_pt.z) / 2.0;
            marker_out.pose.orientation.w = 1.0;

            marker_out.scale.x = size_x;
            marker_out.scale.y = size_y;
            marker_out.scale.z = size_z;

            marker_out.color.r = 0.0;
            marker_out.color.g = 0.5;
            marker_out.color.b = 1.0;
            marker_out.color.a = 0.3;

            return true;
        }

        void clusterObjectCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
        {
            PointCloudPtr cloud(new pcl::PointCloud<pcl::PointXYZ>());
            pcl::fromROSMsg(*msg, *cloud);

            if (cloud->empty())
                return;

            // Euclidean Clustering
            std::vector<pcl::PointIndices> cluster_indices;

            if (!separateGroundAndObstacles(cloud, cluster_indices))
                return;
            
            PointCloudPtr valid_obstacles(new pcl::PointCloud<pcl::PointXYZ>());

            visualization_msgs::msg::MarkerArray marker_array;
            
            // DELETE ALL previous markers
            visualization_msgs::msg::Marker delete_marker;
            delete_marker.action = visualization_msgs::msg::Marker::DELETEALL;
            marker_array.markers.push_back(delete_marker);

            int obstacle_id = 0;

            // Process Each Cluster
            for (const auto& indices : cluster_indices)
            {
                visualization_msgs::msg::Marker marker;
                
                // market id need start from 1 not 0
                if (processSingleCluster(cloud, indices, msg->header, obstacle_id + 1, *valid_obstacles, marker))
                {
                    obstacle_id++;
                    marker_array.markers.push_back(marker);
                }
            }

            // Publish the bounding boxes
            publisher_bbox_->publish(marker_array);
            
            // Publish the valid obstacles point cloud if it's not empty
            if (!valid_obstacles->empty())
            {
                sensor_msgs::msg::PointCloud2 output_msg;
                pcl::toROSMsg(*valid_obstacles, output_msg);
                output_msg.header = msg->header;
                publisher_obstacles_->publish(output_msg);
            }
        }

        // ROS2 Subscribers and Publishers
        rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_object_pcd_;
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_obstacles_;
        rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr publisher_bbox_;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);                                   // Initialize ROS 2
    rclcpp::spin(std::make_shared<LidarClusterObjectNode>());   // Spin the node to process callbacks
    rclcpp::shutdown();
    return 0;
}
