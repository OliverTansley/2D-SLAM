from setuptools import setup
import os
from glob import glob

package_name = 'feature_detection_py'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share',package_name),glob('launch/*launch.[pyx][yma]*'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='u2028576',
    maintainer_email='olivertansley1@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [ 
            ['LineDetector=feature_detection_py.LineDetector']
        ],
    },
)
