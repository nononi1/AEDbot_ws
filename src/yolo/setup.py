from setuptools import find_packages, setup

package_name = 'yolo'

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
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'yoloModel = yolo.yoloModel:main',
            'yoloSJ = yolo.yoloSJ:main',
            'yoloNoUI = yolo.yoloNoUI:main',
            'yoloSJ_modified = yolo.yoloSJ_modified:main',
            'yoloModel_modified = yolo.yoloModel_modified:main',
            'yoloSJ_modified2 = yolo.yoloSJ_modified2:main',
            'yoloModel1 = yolo.yoloModel1:main',
        ],
    },
)
