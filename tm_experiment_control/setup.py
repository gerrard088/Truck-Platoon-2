from setuptools import find_packages, setup

package_name = 'tm_experiment_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/tm_experiment.launch.py', 'launch/deterministic_experiment.launch.py']),
    ],
    install_requires=['setuptools', 'rclpy', 'numpy'],
    zip_safe=True,
    maintainer='tmo',
    maintainer_email='tmo@todo.todo',
    description='Traffic Manager based platoon scenario runner for repeatable comparison experiments.',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'tm_scenario_runner = tm_experiment_control.tm_scenario_runner:main',
            'deterministic_scenario_runner = tm_experiment_control.deterministic_scenario_runner:main',
        ],
    },
)
