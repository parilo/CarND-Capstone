## How to build docker images

### Installing Docker

install docker-ce from [here](https://docs.docker.com/engine/installation/linux/docker-ce/ubuntu/)
```
 # last version issue https://github.com/moby/moby/issues/22207
 # use this version instead of 17.06.0~ce-1~ubuntu
 # or docker may eat all of your disk space
 sudo apt-get install docker-ce=17.06.0~ce-0~ubuntu

 # add yourself tothe docker group and relogin
 sudo usermod -a -G docker username

 # if you don't want to relogin do
 exec su -l username
 # and have shell with already added group
```

### Build an image

Base for docker file was taken from [here](https://github.com/osrf/docker_images/blob/7ba58fc107b368d6409c22161070eb93e562f240/ros/kinetic/ubuntu/xenial/desktop-full/Dockerfile)

# build docker image
```
 # clone this repository
 git clone https://github.com/parilo/CarND-Capstone.git

 # GPU version
 cd CarND-Capstone/docker/gpu
 docker build -t udacity-carnd-ros-gpu .

 # none-GPU version
 cd CarND-Capstone/docker/nogpu
 docker build -t udacity-carnd-ros-nogpu .
```

### Trubleshooting

In case docker fails with apt-get update you may set your dns system wide. [See this](https://robinwinslow.uk/2016/06/23/fix-docker-networking-dns/).
```
 # Create file
 /etc/docker/daemon.json
 # with content
 {
    "dns": ["your dns servers ip", "another ip"]
 }
 # then restart the docker service:
 sudo service docker restart
```
