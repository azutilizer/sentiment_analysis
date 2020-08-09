
## Install depencency

- OS: Ubuntu 18.04
- Python 3.6.7 or later
- OpenVino
- GPU/CPU

```
sudo apt-get update -qq && apt-get install -qy python3 python3-dev python3-pip
sudo apt-get install cmake
sudo apt-get install python3-pyqt5
sudo apt install python3-testresources
pip3 install -r requirements.txt
pip3 install torch torchvision
```

## [OpenCV with CUDA install](https://gist.github.com/raulqf/f42c718a658cddc16f9df07ecc627be7)
- Generic tools:
```
$ sudo apt update
$ sudo apt upgrade
$ sudo apt install build-essential cmake pkg-config unzip yasm git checkinstall
```
- Image I/O libs
```
$ sudo apt install libjpeg-dev libpng-dev libtiff-dev
```
- Video/Audio Libs - FFMPEG, GSTREAMER, x264 and so on
```
$ sudo apt install libavcodec-dev libavformat-dev libswscale-dev libavresample-dev
$ sudo apt install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
$ sudo apt install libxvidcore-dev x264 libx264-dev libfaac-dev libmp3lame-dev libtheora-dev
$ sudo apt install libfaac-dev libmp3lame-dev libvorbis-dev
```
- OpenCore - Adaptive Multi Rate Narrow Band (AMRNB) and Wide Band (AMRWB) speech codec
```
$ sudo apt install libopencore-amrnb-dev libopencore-amrwb-dev
```
- Cameras programming interface libs
```
$ sudo apt-get install libdc1394-22 libdc1394-22-dev libxine2-dev libv4l-dev v4l-utils
$ cd /usr/include/linux
$ sudo ln -s -f ../libv4l1-videodev.h videodev.h
$ cd ~ (cd /datadrive/vision-queue-tracking/py-tracking)
```
- GTK lib for the graphical user functionalites coming from OpenCV highghui module
```
$ sudo apt-get install libgtk-3-dev
```
- others:
```
$ sudo apt-get install libtbb-dev
$ sudo apt-get install libatlas-base-dev gfortran
$ sudo apt-get install libprotobuf-dev protobuf-compiler
$ sudo apt-get install libgoogle-glog-dev libgflags-dev
$ sudo apt-get install libgphoto2-dev libeigen3-dev libhdf5-dev doxygen
```
- [OpenCV] (https://gist.github.com/raulqf/f42c718a658cddc16f9df07ecc627be7)
```
$ wget -O opencv.zip https://github.com/opencv/opencv/archive/4.2.0.zip
$ wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.2.0.zip
$ unzip opencv.zip
$ unzip opencv_contrib.zip

$ echo "Create a virtual environtment for the python binding module"
$ sudo pip3 install virtualenv virtualenvwrapper
$ sudo rm -rf ~/.cache/pip
$ echo "Edit ~/.bashrc"
$ export WORKON_HOME=$HOME/.virtualenvs
$ export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3
$ source /usr/local/bin/virtualenvwrapper.sh
$ mkvirtualenv opencv_cuda -p python3
$ pip install numpy

$ cd opencv-4.2.0
$ mkdir build
$ cd build

$ cmake -D CMAKE_BUILD_TYPE=RELEASE \
	-D CMAKE_C_COMPILER=/usr/bin/gcc-7 \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D INSTALL_PYTHON_EXAMPLES=ON \
    -D INSTALL_C_EXAMPLES=OFF \
    -D WITH_TBB=ON \
    -D WITH_CUDA=ON \
    -D WITH_CUDNN=ON \
    -D OPENCV_DNN_CUDA=ON \
    -D CUDA_ARCH_BIN=6.1 \
    -D BUILD_opencv_cudacodec=OFF \
    -D ENABLE_FAST_MATH=1 \
    -D CUDA_FAST_MATH=1 \
    -D WITH_CUBLAS=1 \
    -D WITH_V4L=ON \
    -D WITH_QT=OFF \
    -D WITH_OPENGL=ON \
    -D WITH_GSTREAMER=ON \
    -D OPENCV_GENERATE_PKGCONFIG=ON \
    -D OPENCV_PC_FILE_NAME=opencv.pc \
    -D OPENCV_ENABLE_NONFREE=ON \
    -D OPENCV_PYTHON3_INSTALL_PATH=~/.virtualenvs/cv/lib/python3.6/site-packages \
    -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib-4.2.0/modules \
    -D PYTHON_EXECUTABLE=~/.virtualenvs/cv/bin/python \
    -D BUILD_EXAMPLES=ON ..
$ make -j6
$ sudo make install

$ sudo /bin/bash -c 'echo "/usr/local/lib" >> /etc/ld.so.conf.d/opencv.conf'
$ sudo ldconfig
```

Alternatively,
```
cmake -D CMAKE_BUILD_TYPE=RELEASE \
	-D CMAKE_C_COMPILER=/usr/bin/gcc-7 \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D INSTALL_C_EXAMPLES=ON \
    -D WITH_TBB=ON \
    -D WITH_CUDA=ON \
    -D WITH_CUDNN=ON \
    -D OPENCV_DNN_CUDA=ON \
    -D CUDA_ARCH_BIN=6.1 \
    -D BUILD_opencv_cudacodec=ON \
    -D ENABLE_FAST_MATH=1 \
    -D CUDA_FAST_MATH=1 \
    -D WITH_CUBLAS=1 \
    -D WITH_V4L=ON \
    -D WITH_QT=OFF \
    -D WITH_OPENGL=ON \
    -D WITH_GSTREAMER=ON \
    -D OPENCV_GENERATE_PKGCONFIG=ON \
    -D OPENCV_PC_FILE_NAME=opencv.pc \
    -D OPENCV_ENABLE_NONFREE=ON \
    -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib-4.2.0/modules \
    -D PYTHON_EXECUTABLE=/usr/bin/python3 \
    -D BUILD_EXAMPLES=ON ..
```

## OpenCV dldt build
```
$ git clone https://github.com/opencv/dldt.git
$ cd dldt/inference-engine
$ git submodule init
$ git submodule update --recursive
$ cd ..
$ mkdir build
$ cd build
$ cmake -DCMAKE_BUILD_TYPE=Release -DENABLE_PYTHON=ON -DPYTHON_EXECUTABLE=/usr/bin/python3.6 -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so -DPYTHON_INCLUDE_DIR=/usr/include/python3.6 -DENABLE_MKL_DNN=ON -DENABLE_CLDNN=ON ..
$ make -j6

```
For CMake projects, set an environment variable InferenceEngine_DIR:

    export InferenceEngine_DIR=/path/to/dldt/inference-engine/build/

     
## Openvino install
1. First install [OpenVINO](https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_linux.html).
- Download the distribution of OpenVINO toolkit for Linux* package file to current user's Downloads directory.
- cd ~/Downloads/
- tar -zxvf l_openvino_toolkit_p_2020.1.023.tgz
- cd l_openvino_toolkit_p_2020.1.023
- sudo ./install.sh
2. After it's done, you need to install dependencies.
- cd /opt/intel/openvino/install_dependencies
- sudo -E ./install_openvino_dependencies.sh
3. Set the Environment Variables
- source /opt/intel/openvino/bin/setupvars.sh
- sudo nano ~/.bashrc

Add this line to the end of the file:

- source /opt/intel/openvino/bin/setupvars.sh

Save and close the file: Ctrl+O, Ctrl+X (nano), Esc and :wq (vim)
- source ~/.bashrc

4. Check installing
```
$ cd ~/intel/openvino/deployment_tools/demo
$ ./demo_squeezenet_download_convert_run.sh
$ ./demo_security_barrier_camera.sh
```
When the verification script completes, you will see an image that displays the resulting frame with detections rendered as bounding boxes, and text.

Close the image viewer window to complete the verification script.


Run this script;

    Usage:
	    python3 face_emo_recognizer.py --input <test.jpg>


# Solutions

## 1. Correction in OpenCV's default CMAKE search Path for OpenBLAS Library on Ubuntu-64bit Machines

OpenBLAS can installed from respected default repository with the following command:
```
sudo apt install libopenblas-dev, libopenblas-base
```
Whereas LAPACK Library is little trickier to install, as OpenBLAS doesn't provide Lapacke header file(i cannot find the lapacke.h header fie ), Therefore I ought to install LAPACK Library instead, which works absolutely fine:
```
sudo apt install liblapacke-dev
sudo ln -s /usr/include/lapacke.h /usr/include/x86_64-linux-gnu # corrected path for the library 
```

### Setting Correct Path:

So, The only thing left is setting Correct Default Search Path for Include and Libs Directory of OpenBLAS Library which is as below :

- Open_BLAS_INCLUDE_SEARCH_PATHS path: /usr/include/x86_64-linux-gnu
- Open_BLAS_LIB_SEARCH_PATHS path: /usr/lib/x86_64-linux-gnu

Appending these respective additional Paths in OpenCVFindOpenBLAS.cmake :
```
opencv/cmake/OpenCVFindOpenBLAS.cmake
    SET(Open_BLAS_INCLUDE_SEARCH_PATHS 
```
and
```
opencv/cmake/OpenCVFindOpenBLAS.cmake
    SET(Open_BLAS_LIB_SEARCH_PATHS 
```
Completely solves the issue without altering any additional file in OpenCV.
