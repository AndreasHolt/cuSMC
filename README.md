# SMAcc BB

This project is a 7th Semester Project for computer science Aalborg University

SMAcc BB is a statistical model checking tool, that utilizes Cuda to speed up computation.

## Table of Contents
- [About the Project](About-the-Project)
- [How to run](How-to-run)
- [Requirements](Requirements)
- [Contributors](Contributors)
## About the Project
This program builds upon and seeks to compare itself with the [SMAcc implementation](https://github.com/Baksling/P7-SMAcc), which runs simulations in different threads.
Our implementation, SMAcc but better (SMAcc BB), instead runs a thread for each component of each simulation. Thus speeding up computation (we hope).

## How to run
This program uses cmake.lists to link and compile the program. The program can then simply be run. During development we utilized CLion from JetBrains 

To alter run configrations refer to the [main file](main.cu)

## Requirements
This program utilizes the Nvidia Cuda toolset for C++ and has been tested on windows and linux

## Contributors

This project is made by:
- Andreas Holm
- Daniel Hansen
- Grace Melchiors
- Mikkel Bjørn
- Hjalte Johnson
- Theis Mathiassen
