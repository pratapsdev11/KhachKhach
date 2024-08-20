from setuptools import setup, find_packages

setup(
    name='khach_khach',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'opencv-python',
        'numpy',
        'ultralytics',
    ],
    author='pratapdevs11',
    author_email='divyapratap360@gmail.com',
    description='A package for processing video frames, annotating keypoints, and more.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/pratapsdev11/Khach_Khach/tree/main',  # Update with your URL
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
