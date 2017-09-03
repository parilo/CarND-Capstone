FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04

# copined from
# https://github.com/osrf/docker_images/blob/7ba58fc107b368d6409c22161070eb93e562f240/ros/kinetic/ubuntu/xenial/ros-core/Dockerfile

# install packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    dirmngr \
    gnupg2 \
    && rm -rf /var/lib/apt/lists/*

# setup keys
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 421C365BD9FF1F717815A3895523BAEEB01FA116

# setup sources.list
RUN echo "deb http://packages.ros.org/ros/ubuntu xenial main" > /etc/apt/sources.list.d/ros-latest.list

# install bootstrap tools
RUN apt-get update && apt-get install --no-install-recommends -y \
    python-rosdep \
    python-rosinstall \
    python-vcstools \
    && rm -rf /var/lib/apt/lists/*

# setup environment
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# bootstrap rosdep
RUN rosdep init \
    && rosdep update

# install ros packages
ENV ROS_DISTRO kinetic
RUN apt-get update && apt-get install -y \
    ros-kinetic-ros-core=1.3.1-0* \
    && rm -rf /var/lib/apt/lists/*

# setup entrypoint
COPY ./ros_entrypoint.sh /

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]

RUN apt-get update && apt-get install -y \
    ros-kinetic-desktop-full=1.3.1-0* \
    ros-kinetic-desktop=1.3.1-0* \
    ros-kinetic-robot=1.3.1-0* \
    ros-kinetic-ros-base=1.3.1-0* \
    && rm -rf /var/lib/apt/lists/*



# udacity carnd part

RUN apt-get update && apt-get install -y \
    python-pip \
    sudo wget mc netbase \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

ADD requirements.txt requirements.txt

RUN pip install -r requirements.txt

RUN echo "%sudo ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

RUN mkdir /home/ros
RUN chown root:sudo /home/ros
RUN chmod 770 /home/ros

# probably your host user will have one of that uids
RUN useradd -r -u 1000 -d /home/ros -G sudo carnd1
RUN useradd -r -u 1001 -d /home/ros -G sudo carnd2
RUN useradd -r -u 1002 -d /home/ros -G sudo carnd3
RUN useradd -r -u 1003 -d /home/ros -G sudo carnd4
RUN useradd -r -u 1004 -d /home/ros -G sudo carnd5
RUN useradd -r -u 1005 -d /home/ros -G sudo carnd6
RUN useradd -r -u 1006 -d /home/ros -G sudo carnd7
RUN useradd -r -u 1007 -d /home/ros -G sudo carnd8
RUN useradd -r -u 1008 -d /home/ros -G sudo carnd9
RUN useradd -r -u 1009 -d /home/ros -G sudo carnd10
RUN useradd -r -u 1010 -d /home/ros -G sudo carnd11
RUN useradd -r -u 1011 -d /home/ros -G sudo carnd12
RUN useradd -r -u 1012 -d /home/ros -G sudo carnd13
RUN useradd -r -u 1013 -d /home/ros -G sudo carnd14
RUN useradd -r -u 1014 -d /home/ros -G sudo carnd15
RUN useradd -r -u 1015 -d /home/ros -G sudo carnd16

USER carnd1

ADD install_dbw_mkz.sh install_dbw_mkz.sh
RUN sudo -u carnd1 bash install_dbw_mkz.sh

EXPOSE 4567
