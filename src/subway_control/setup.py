from setuptools import setup
import os
from glob import glob

package_name = 'subway_control'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # 템플릿 파일(html) 포함 설정
        (os.path.join('share', package_name, 'templates'), glob('subway_control/templates/*.html')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='SafetyManager',
    maintainer_email='admin@subway.com',
    description='Subway Safety Control System UI',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'start = subway_control.control_tower:main',
        ],
    },
)
