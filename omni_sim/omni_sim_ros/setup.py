from setuptools import find_packages, setup
import os
from glob import glob

package_name = "omni_sim_ros"

setup(
    name=package_name,
    version="1.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages",
         ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        (os.path.join("share", package_name, "rviz"), glob("rviz/*.rviz")),
    ],
    install_requires=["setuptools", "omni_sim_core"],
    zip_safe=True,
    maintainer="maeda",
    maintainer_email="riko26maeda@gmail.com",
    description="ROS 2 wrapper around omni_sim_core.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "sim_node = omni_sim_ros.sim_node:main",
        ],
    },
)
