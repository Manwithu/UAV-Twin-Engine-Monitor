# UAV-Twin-Engine-Monitor
This is a project explains method to extract data from sensors and implement as a twin engine monitor software.
# Arduino Script
Download the given sketch and edit the sensors to your preferences . Here I have used flame sensors as an example which detect heat and transmits the message through serial monitor.
This can be used in a microcomputer like Rasberry Pi connected to NodeMCU.
# Use Of Python Program
The python program is used to log the sensor outputs along with their CAN IDs which can be transferred to another device connected to the network in CSV format.

Even though due to hardware limitation , the original CAN protocol cannot be processed , the given code imitates the CAN networking protocol.
Learn more about CAN ( Controller Area Networking) in linux at https://docs.kernel.org/networking/can.html
