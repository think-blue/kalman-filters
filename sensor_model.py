import knowledge as knowledge
import math as math
import numpy as np

"""
This is the sensor model
Used to calculate the position of the robot using triangulation
This model does not consider beacons being blocked off by walls for simplicity.
"""

PRECISION = 2;
SENSOR_RANGE = 60;

class Sensor_Model:
    def __init__(self, sensor_range = SENSOR_RANGE, beacon_knowledge = knowledge.Beacon):
        print("Sensor activated");
        self.beacon_knowledge = beacon_knowledge; #knowledge.Beacon;
        self.sensor_range = sensor_range;

    def scan_environment(self, sim_robot_pos, sim_theta):
        """
        :param robot_pos: This is a tuple of (x,y) which specifies the current position of the robot
        :param sensor_range: It is a range of the area that the sensor covers or scan for beacons.
        :return: returns a list of tuples. Each tuple consists of beacon enum and distance that the sensor model has found
        """
        visible_beacons = [];
        for beacon in self.beacon_knowledge:
            if(self.get_distance(beacon.value,sim_robot_pos)<=self.sensor_range):
                visible_beacons.append(tuple((beacon,self.get_distance(beacon.value,sim_robot_pos))));
        return self.triangulate(visible_beacons, sim_theta);


    def get_distance(self, p1, p2):
        """

        :param p1: tuple (x1,y1)
        :param p2: tuple (x2,y2)
        :return: Returns the cartesian distance between p1 and p2
        """
        x1, y1 = p1[0], p1[1];
        x2, y2 = p2[0], p2[1];

        return math.sqrt(math.pow((x2 - x1), 2) + math.pow((y2 - y1), 2));


    def triangulate(self, beacons_info, sim_theta):
        """
        Ref for this algorithm: https://goo.gl/EfjWs1 and https://goo.gl/2tvmYQ
        :param beacons_info: Is a list of tuples. Each tuple consists of beacon enum and distance
        :return: The calculated x,y coordinate of the robot
        """
        print("No: beacons:"+str(beacons_info.__len__()));
        intersect_points = [];

        for i in range(beacons_info.__len__() - 1):

            if (intersect_points.__len__() == 0): # No intersect points
                p0, r0 = beacons_info[i][0].value, beacons_info[i][1];
                p1, r1 = beacons_info[i + 1][0].value, beacons_info[i + 1][1];

                D = self.get_distance(p0, p1);

                a,b = p0[0],p0[1];
                c,d = p1[0],p1[1];

                delta = (1 / 4) * math.sqrt((D + r0 + r1) * (D + r0 - r1) * (D - r0 + r1) * (-D + r0 + r1));

                x1 = round(((a+c)/2 + ((c-a)*(r0**2 - r1**2))/(2*D**2) +(2*(b-d)*delta)/(D**2)),PRECISION) + 0; #+0 is used to make negative zeros to positive zeros, if we don't set the precision we will get value like 0.2*e-30 or s
                x2 = round(((a+c)/2 + ((c-a)*(r0**2 - r1**2))/(2*D**2) -(2*(b-d)*delta)/(D**2)),PRECISION) + 0;

                y1 = round(((b+d)/2+((d-b)*(r0*r0-r1*r1))/(2*D*D)-(2*(a-c)*delta)/(D*D)),PRECISION) + 0;
                y2 = round(((b+d)/2+((d-b)*(r0*r0-r1*r1))/(2*D*D)+(2*(a-c)*delta)/(D*D)), PRECISION) + 0;

                intersect_points.append(tuple((x1,y1)));
                intersect_points.append(tuple((x2,y2)));

            else: # contains intersect points

                p = beacons_info[i + 1][0].value;
                r = beacons_info[i+1][1];

                for intersect_point in intersect_points:
                    distance = self.get_distance(p, intersect_point);
                    if(not(distance>=r-1 and distance<=r+1)):  #if ( distance!= r+0.5 or)
                        intersect_points.remove(intersect_point);

        if(intersect_points.__len__()!=1):
            x = (intersect_points[0][0] + intersect_points[1][0])/2;
            y = (intersect_points[0][1] + intersect_points[1][1])/2;
            print("Multiple intersections");
        else:
            x = intersect_points[0][0];
            y = intersect_points[0][1];
            print("Single Intersection")

        return np.array([x,y,sim_theta]);

