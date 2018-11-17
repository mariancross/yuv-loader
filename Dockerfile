FROM floopcz/tensorflow_cc:archlinux-shared-cuda

# switch the url to a good rating mirror when running this. Check https://www.archlinux.org/mirrors/status/
RUN echo 'Server = http://mirrors.udenar.edu.co/archlinux/$repo/os/$arch' > /etc/pacman.d/mirrorlist

# install the opencv dependencies before (including build dependencies)
RUN pacman -Sy intel-tbb openexr gst-plugins-base libdc1394 cblas lapack libgphoto2 jasper ffmpeg cmake python-numpy python2-numpy mesa eigen hdf5 lapacke gtk3 vtk glew --noconfirm

RUN pacman -Sy opencv --noconfirm

COPY . /opt/yuv-loader

RUN cd /opt/yuv-loader && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make
    