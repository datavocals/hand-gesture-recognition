#### Project name

Hand-gesture visual cruise controller

#### Motivation

As the ground vehicle become much more 'smart', we can see much more vision-controlled systems are applied. It's really a great opportunity to explore an application that helps the vehicle to be smarter.

#### Objective

Use jetson nano platform to do the hand-gesture recognization and the arduino controls the speed of the DC motor using PID. 

The reference speed is set by the the figure numbers shown by the  hand gesture. There will be five levels of speeds.

These two platforms will communicate via serial port.

#### Method

The major challenge of this project is how to train the model and do the gesture recognization. We will capture the images of the hand gesture and use these samples to train the model.

