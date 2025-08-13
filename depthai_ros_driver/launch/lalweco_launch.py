
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
	pkg_share = get_package_share_directory('depthai_ros_driver')
	default_config = os.path.join(pkg_share, 'config', 'lalweco.yaml')

	return LaunchDescription([
		DeclareLaunchArgument(
			'params_file',
			default_value=default_config,
			description='Path to the parameter file to use.'
		),
		DeclareLaunchArgument(
			'camera_name',
			default_value='oak',
			description='Camera node name.'
		),
		Node(
			package='depthai_ros_driver',
			executable='camera_node',
			name=LaunchConfiguration('camera_name'),
			output='screen',
			parameters=[LaunchConfiguration('params_file')],
		)
	])
