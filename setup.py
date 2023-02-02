from setuptools import setup, find_packages

setup(
    name="LPRNet",
    version="0.0.2",
    author="Heonjin Kwon",
    author_email="kwon@4ind.co.kr",
    description="A Package for license plate recognition written in pytorch-lightning",
    keywords=['pytorch', 'pytorch-lightning', 'license-plate-recognition'],
    install_requires=[
        'pytorch-lightning>=1.7.0, <=1.9.0',
        'numpy>=1.17.1',
        # 'imutils>=0.4.0',
        'natsort==8.2.0',
        'lmdb==1.4.0',
        'Pillow==9.4.0',
        'nltk==3.8.1'
    ],
    packages=find_packages(),
    classifiers=[ 
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent"
    ],
    python_requires='>=3.7',   
)
