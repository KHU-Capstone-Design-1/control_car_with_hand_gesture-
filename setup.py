from setuptools import setup

package_name = 'hand_tracking_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='yangjunseok',
    maintainer_email='yangjunseok@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'hand_tracking_node = hand_tracking_pkg.hand_tracking_node:main',
            'hand_tracking_node2 = hand_tracking_pkg.hand_tracking_node2:main',
            'img_publisher = hand_tracking_pkg.img_publisher:main',
            'img_subscriber = hand_tracking_pkg.img_subscriber:main',
        ],
    },
)


