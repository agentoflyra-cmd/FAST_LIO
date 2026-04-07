#include <functional>
#include <pcl/impl/point_types.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/logging.hpp>
#include <rclcpp/node.hpp>
#include <rclcpp/publisher.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/subscription.hpp>
#include <rclcpp/timer.hpp>
#include <sensor_msgs/msg/detail/point_cloud2__struct.hpp>
#include <string>
#include <pcl/registration/ndt.h>
#include <pcl/registration/icp.h>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <pcl/filters/voxel_grid.h>
#include <std_srvs/srv/trigger.hpp>
#include <vector>
#include <Eigen/Geometry>
#include <algorithm>

#define MAT4F_FROM_ARRAY(v)                                                                        \
    v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], v[11], v[12], v[13], v[14], \
        v[15]

class Relocalization : public rclcpp::Node
{
  public:
    Relocalization() : Node("relocalization_node")
    {
        this->declare_parameter<std::string>("map_file_path", "test.pcd");
        this->declare_parameter<std::vector<double>>(
            "localization.initial_guess", {1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.});
        this->declare_parameter<double>("localization.icp_max_corr_dist", 1.0);
        this->declare_parameter<int>("localization.icp_max_iterations", 40);
        this->declare_parameter<double>("localization.icp_trans_eps", 1e-3);
        this->declare_parameter<double>("localization.icp_fitness_threshold", 0.5);
        this->declare_parameter<double>("localization.tracking_icp_max_corr_dist", 0.5);
        this->declare_parameter<int>("localization.tracking_icp_max_iterations", 20);
        this->declare_parameter<double>("localization.tracking_icp_fitness_threshold", 0.3);

        this->get_parameter_or<std::string>("map_file_path", map_path, "./test.pcd");
        std::vector<double> raw_vector;
        this->get_parameter_or("localization.initial_guess", raw_vector,
                               {1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.});
        if (raw_vector.size() != 16)
        {
            throw std::runtime_error("localization.initial_guess must contain 16 elements");
        }
        map_cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>);
        initial_guess_ << MAT4F_FROM_ARRAY(raw_vector);
        current_guess_ = initial_guess_;
        this->get_parameter_or<double>("localization.icp_max_corr_dist", icp_max_corr_dist_, 1.0);
        this->get_parameter_or<int>("localization.icp_max_iterations", icp_max_iterations_, 40);
        this->get_parameter_or<double>("localization.icp_trans_eps", icp_trans_eps_, 1e-3);
        this->get_parameter_or<double>("localization.icp_fitness_threshold", icp_fitness_threshold_, 0.5);
        this->get_parameter_or<double>("localization.tracking_icp_max_corr_dist", tracking_icp_max_corr_dist_, 0.5);
        this->get_parameter_or<int>("localization.tracking_icp_max_iterations", tracking_icp_max_iterations_, 20);
        this->get_parameter_or<double>("localization.tracking_icp_fitness_threshold", tracking_icp_fitness_threshold_, 0.3);

        if (pcl::io::loadPCDFile(map_path, *map_cloud_) < 0)
        {
            RCLCPP_FATAL(this->get_logger(), "Failed to load map PCD file!");
            throw std::runtime_error("Map load failed");
        }

        ndt_.setResolution(1.0);
        ndt_.setMaximumIterations(50);
        ndt_.setTransformationEpsilon(0.01);
        ndt_.setStepSize(0.1);
        ndt_.setInputTarget(map_cloud_);
        icp_.setInputTarget(map_cloud_);
        icp_.setMaxCorrespondenceDistance(icp_max_corr_dist_);
        icp_.setMaximumIterations(icp_max_iterations_);
        icp_.setTransformationEpsilon(icp_trans_eps_);

        RCLCPP_INFO(this->get_logger(), "Map load: %zu points", this->map_cloud_->size());
        map_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/offline_map", 1);
        // publish_map();
        timer_ = this->create_wall_timer(
            std::chrono::seconds(static_cast<int64_t>(500.0 / 100.0)),
            std::bind(&Relocalization::publish_map, this));
        cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/cloud_registered", 1,
            std::bind(&Relocalization::cloud_callback, this, std::placeholders::_1));
        map_to_odom_pub_ = this->create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>(
            "/map_to_odom", 1);
        reset_srv_ = this->create_service<std_srvs::srv::Trigger>(
            "/relocalization_reset",
            std::bind(&Relocalization::reset_callback, this, std::placeholders::_1, std::placeholders::_2));

    }

    void publish_map() {
        // RCLCPP_INFO(this->get_logger(), "offline map publishing...");
        auto msg = sensor_msgs::msg::PointCloud2();
        pcl::toROSMsg(*map_cloud_, msg);
        msg.header.stamp = now();
        msg.header.frame_id = "map";
        map_pub_->publish(msg);
    }

    void cloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr current_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*msg, *current_cloud);
        if (current_cloud->empty())
        {
            RCLCPP_WARN(this->get_logger(), "Received empty point cloud");
            return;
        }

        pcl::VoxelGrid<pcl::PointXYZ> vf;
        vf.setLeafSize(0.3f, 0.3f, 0.3f);
        vf.setInputCloud(current_cloud);
        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        vf.filter(*filtered_cloud);
        if (filtered_cloud->empty())
        {
            RCLCPP_WARN(this->get_logger(), "Filtered point cloud is empty");
            return;
        }

        if (tracking_active_)
        {
            if (track_with_icp(filtered_cloud))
            {
                return;
            }

            RCLCPP_WARN(this->get_logger(), "Tracking ICP lost, falling back to coarse relocalization.");
            tracking_active_ = false;
            relocalized_ = false;
        }

        ndt_.setResolution(1.0);
        ndt_.setInputSource(filtered_cloud);
        pcl::PointCloud<pcl::PointXYZ> aligned;
        ndt_.align(aligned, current_guess_);
        if (!ndt_.hasConverged() || ndt_.getFitnessScore() > 1.0)
        {
            RCLCPP_WARN(get_logger(), "NDT coarse match failed to converge, fitness=%.3f",
                        ndt_.getFitnessScore());
            return;
        }
        Eigen::Matrix4f T_coarse = ndt_.getFinalTransformation();

        ndt_.setResolution(0.3);
        ndt_.align(aligned, T_coarse);
        if (!ndt_.hasConverged() || ndt_.getFitnessScore() > 1.0)
        {
            RCLCPP_WARN(get_logger(), "NDT fine match failed to converge, fitness=%.3f",
                        ndt_.getFitnessScore());
            return;
        }
        Eigen::Matrix4f T_ndt = ndt_.getFinalTransformation();

        icp_.setInputSource(filtered_cloud);
        pcl::PointCloud<pcl::PointXYZ> icp_aligned;
        icp_.align(icp_aligned, T_ndt);
        if (!icp_.hasConverged() || icp_.getFitnessScore() > icp_fitness_threshold_)
        {
            RCLCPP_WARN(
                this->get_logger(),
                "ICP refine failed, fitness=%.3f threshold=%.3f",
                icp_.getFitnessScore(),
                icp_fitness_threshold_);
            return;
        }

        Eigen::Matrix4f T = icp_.getFinalTransformation();
        current_guess_ = T;
        tracking_active_ = true;
        relocalized_ = true;
        RCLCPP_INFO(
            this->get_logger(),
            "Relocalization success, NDT fitness: %.3f, ICP fitness: %.3f",
            ndt_.getFitnessScore(),
            icp_.getFitnessScore());
        publish_map_to_odom(T, icp_.getFitnessScore());
    }
    

    void publish_map_to_odom(const Eigen::Matrix4f & T, double fitness_score)
    {
        Eigen::Matrix3f R = T.block<3, 3>(0, 0);
        Eigen::Quaternionf q(R);
        q.normalize();

        auto pose_msg = geometry_msgs::msg::PoseWithCovarianceStamped();
        pose_msg.header.stamp = now();
        pose_msg.header.frame_id = "map";
        pose_msg.pose.pose.position.x = T(0, 3);
        pose_msg.pose.pose.position.y = T(1, 3);
        pose_msg.pose.pose.position.z = T(2, 3);
        pose_msg.pose.pose.orientation.w = q.w();
        pose_msg.pose.pose.orientation.x = q.x();
        pose_msg.pose.pose.orientation.y = q.y();
        pose_msg.pose.pose.orientation.z = q.z();

        double cov = std::max(1e-3, fitness_score * 0.1);
        pose_msg.pose.covariance[0] = cov;
        pose_msg.pose.covariance[7] = cov;
        pose_msg.pose.covariance[14] = cov;
        pose_msg.pose.covariance[21] = cov;
        pose_msg.pose.covariance[28] = cov;
        pose_msg.pose.covariance[35] = cov * 2.0;
        map_to_odom_pub_->publish(pose_msg);
    }

    void reset_callback(
        const std::shared_ptr<std_srvs::srv::Trigger::Request> req,
        std::shared_ptr<std_srvs::srv::Trigger::Response> res)
    {
        (void)req;
        relocalized_ = false;
        tracking_active_ = false;
        current_guess_ = initial_guess_;
        RCLCPP_WARN(this->get_logger(), "Relocalization state reset by service request.");
        res->success = true;
        res->message = "Relocalization reset.";
    }

    bool track_with_icp(const pcl::PointCloud<pcl::PointXYZ>::Ptr &filtered_cloud)
    {
        pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> tracking_icp;
        tracking_icp.setInputTarget(map_cloud_);
        tracking_icp.setInputSource(filtered_cloud);
        tracking_icp.setMaxCorrespondenceDistance(tracking_icp_max_corr_dist_);
        tracking_icp.setMaximumIterations(tracking_icp_max_iterations_);
        tracking_icp.setTransformationEpsilon(icp_trans_eps_);

        pcl::PointCloud<pcl::PointXYZ> aligned;
        tracking_icp.align(aligned, current_guess_);
        if (!tracking_icp.hasConverged() ||
            tracking_icp.getFitnessScore() > tracking_icp_fitness_threshold_)
        {
            RCLCPP_WARN(
                this->get_logger(),
                "Tracking ICP failed, fitness=%.3f threshold=%.3f",
                tracking_icp.getFitnessScore(),
                tracking_icp_fitness_threshold_);
            return false;
        }

        current_guess_ = tracking_icp.getFinalTransformation();
        publish_map_to_odom(current_guess_, tracking_icp.getFitnessScore());
        return true;
    }

    

  private:
    std::string map_path;
    pcl::PointCloud<pcl::PointXYZ>::Ptr map_cloud_;

    pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt_;
    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp_;

    Eigen::Matrix4f initial_guess_;
    Eigen::Matrix4f current_guess_;
    double icp_max_corr_dist_{1.0};
    int icp_max_iterations_{40};
    double icp_trans_eps_{1e-3};
    double icp_fitness_threshold_{0.5};
    double tracking_icp_max_corr_dist_{0.5};
    int tracking_icp_max_iterations_{20};
    double tracking_icp_fitness_threshold_{0.3};
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr map_to_odom_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr map_pub_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr reset_srv_;
    rclcpp::TimerBase::SharedPtr timer_;
    bool relocalized_{false};
    bool tracking_active_{false};
};

int main(int argc, char ** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<Relocalization>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
