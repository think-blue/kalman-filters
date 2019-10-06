
"""
This represents the map or the knowledge that the robot has about the environment. B1,B2, B3,... etc represents the beacon position info.
"""
from enum import Enum

class Beacon(Enum):
    B1 = (0,0);
    B2 = (0,100);
    B3 = (100,0);
    B4 = (100,100);

    #More beacons

    B5 = (40,60);#
    #B6 = (70,80);
    B7 = (40,40);#
    B8 = (40,70);#
    B9 = (60,80);#Currently beacon is drawn in gui in 60,80 but it could actually be 70,80... have to confirm it.



#print(Beacon.B1.value[0]);




