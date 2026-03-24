from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    num_trucks = LaunchConfiguration('num_trucks')
    map_name = LaunchConfiguration('map')
    experiment_tag = LaunchConfiguration('experiment_tag')

    bridge_launch = os.path.join(
        get_package_share_directory('carla-virtual-platoon'),
        'launch',
        'carla-virtual-platoon.launch.py',
    )

    bridge = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(bridge_launch),
        launch_arguments={
            'NumTrucks': num_trucks,
            'Map': map_name,
            'control_enabled': 'false',
        }.items(),
    )

    runner = Node(
        package='tm_experiment_control',
        executable='tm_scenario_runner',
        name='tm_scenario_runner',
        output='screen',
        additional_env={
            'TM_EXPERIMENT_TAG': experiment_tag,
        },
    )

    return LaunchDescription([
        DeclareLaunchArgument('num_trucks', default_value='3'),
        DeclareLaunchArgument('map', default_value='Town04_Opt'),
        DeclareLaunchArgument('experiment_tag', default_value='tm_experiment'),
        bridge,
        runner,
    ])
