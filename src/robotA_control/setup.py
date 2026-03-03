from setuptools import find_packages, setup

package_name = 'robotA_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jaewookim',
    maintainer_email='kjwmechacont@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [

            'robotA_control = robotA_control.robotA_control:main',
            'robotA_control_with_YOLO = robotA_control.robotA_control_with_YOLO:main',
            'robotA_control_with_YOLO_1 = robotA_control.robotA_control_with_YOLO_1:main',
            'robotA_control_with_YOLO_2 = robotA_control.robotA_control_with_YOLO_2:main',
            'robotA_control_with_YOLO_3 = robotA_control.robotA_control_with_YOLO_3:main',
            'robotA_control_with_YOLO_4 = robotA_control.robotA_control_with_YOLO_4:main',
            'robotB_control = robotA_control.robotB_control:main',
            'robotB_control_1 = robotA_control.robotB_control_1:main',

        ],
    },
)
