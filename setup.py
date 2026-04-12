from setuptools import find_packages, setup


setup(
    name="spine",
    version="0.1.0",
    description="SPINE model package",
    packages=find_packages(),
    include_package_data=False,
    package_data={
        "spine.app.preprocessing": ["README.md"],
        "spine.io_utils": [
            "local_ckpts.json",
            "pretrained_configs/*.json",
        ],
    },
)
