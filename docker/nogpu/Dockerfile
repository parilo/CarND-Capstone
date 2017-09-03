FROM osrf/ros:kinetic-desktop-xenial

# install ros packages
RUN apt-get update && apt-get install -y \
    ros-kinetic-desktop-full=1.3.1-0* \
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
