from glob import glob
from setuptools import setup, find_packages

package_name = 'truck_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/matlab', glob('matlab/*.m')),
    ],
    install_requires=[
        'setuptools',
        'rclpy',
        'numpy',
        'opencv-python'
    ],
    zip_safe=True,
    maintainer='tmo',
    maintainer_email='tmo@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'command_publisher = truck_control.command_publisher:main',
            'lane_following_node = truck_control.lane_following_node:main',
            'lane_detect = truck_control.lane_detect:main',
            'pid_controller = truck_control.pid_controller:main',
            'distance_sensor = truck_control.distance_sensor:main',
            'platooning_manager = truck_control.platooning_manager:main',
            'v2v_comm = truck_control.v2v_comm:main',
            'ui_k = truck_control.ui_k:main',
            'ui_tkinter = truck_control.ui_tkinter:main',
            'carla_spectator_follower = truck_control.carla_spectator_follower:main',
            'energy_soc_bridge = truck_control.energy_soc_bridge:main',
            'energy_monitor = truck_control.energy_monitor:main',
            'energy_dashboard = truck_control.energy_dashboard:main',
            'soc_cycle_dashboard = truck_control.soc_cycle_dashboard:main',
            'set_location = truck_control.set_location:main',
        ],
    },
)
