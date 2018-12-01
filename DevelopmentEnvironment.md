# Development Environment FAQ
## General Tips and Tricks
### Q: If you are having trouble with latency when turning on camera images in the simulator, 
A: try classifying only every third or fourth camera image.
### Q: If you are uploading your own previous work, 
A: be sure to make all Python files executable. In Ubuntu, this can be done from the command line with the chmod command. The following command should add executable permissions to all Python files in the specified directory:
```
find /home/workspace/your/directory -type f -iname "*.py" -exec chmod +x {} \;
```

# ROS and Similator an the same machine
## Linux: Udacity Workspace
### Q: On macOS High Sierra, the Web Desktop doesn't open in Safari 12.0.1
A: Use Google Chrome instead
### Q: if you are uploading your own previous work, 
A: you will need to add the folder found in /home/workspace/CarND-Capstone/ros/src/dbw_mkz_msgs to your project.


# ROS and simulator on diofferent machines
## ROS in Udacity Linux 16.04 Linux VM, Simulator natively on MacOS
If you have two hosts available you can distribute the load.

### Q: How to setup communication between simulator and ROS on another PC or in a VM
A: The simulator expects the ROS Server to listen on port 4567 on the localhost. If you want to move ROS to another host, you need to forward port 4567 of the remote host to localhost. (thanks to [nsanghi's post on slack](https://carnd.slack.com/archives/C6NVDVAQ3/p1504363170000027?thread_ts=1504354174.000031)
  1. install openssh server on ubuntu (Ubuntu terminal)
     ```
     sudo apt install openssh-server
     ```
  1. no chnage in sshd_config file. Still started/restarted the service. (Ubuntu terminal)
     ```
     sudo systemctl restart sshd.service
     ```
  1. created key on ubuntu. Accepted all defaults and did not create a passphrase (Ubuntu terminal)
     ```
     ssh-keygen
     ```
  1. enabled ssh sharing on mac (mac)
     System Pre -> sharing -> remote Login (check this box)
  1. copied public key from ubuntu to mac. ran this on ubunutu. (Ubuntu terminal)
     ```
     ssh-copy-id <mac-user>@<mac-ip>
     ```
  1. checked if I can login from ubuntu to mac. Ran this on ubuntu (Ubuntu terminal)
     ```
     ssh '<mac-user>@<mac-ip>'
     ```
  1. logged out(Ubuntu terminal)
     ```
     exit
     ```
  1. created key on mac. Accepted all the defaults and did not create a passphrase(mac terminal)
     ```
     ssh-keygen
     ```
  1. copied public key from mac to ubuntu. ran this on mac. (mac terminal)
     ```
     ssh-copy-id <ubuntu-user>@<ubuntu-ip>
     ```
  1. checked if I can login from mac to ubuntu. Ran this on mac (mac terminal)
     ```
     ssh '<ubuntu-user>@<ubuntu-ip>'
     ```
  1. logged out(mac terminal)
     ```
     exit
     ```
  1. Setup ssh tunnel on mac (mac terminal)
     ```
     ssh -L 4567:127.0.0.1:4567 <ubuntu-user>@<ubuntu-ip>
     ```
  1. left the terminal open
  1. Now I can run ROS on ubuntu and Simulator on mac

### Q: When connecting to ROS that is running on Linux in a VM I am getting the following error message: `"GET /socket.io/?EIO=4&transport=websocket HTTP/1.1" 404`
A: update python SocketIO to version 2.1.0 as recommended by https://github.com/blown302/CarND-Capstone/commit/dfb2166bccbfc6cd5226bb4fe31c0e3b20d2bc94#diff-e4083f2d74d0aa28cb6ce89d0a8a9569. 
