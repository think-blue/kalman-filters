import ea as ea
import robotic_system as rs #the nn and the robotic model is here
import numpy as np
import math as math
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
import time
import sensor_model as sm;


POPULATION_SIZE = 80
SELECTION_COUNT = 50
MUTATION_PROBABILITY = 0.2
SPLIT_POINT = 26


def calculateFitness(velocity_vector):
    """ Calculates fitness based on the velocity vector and output of
        motion sensor """
    translational_motion_factor = abs(1 - math.sqrt( abs( velocity_vector[0] - velocity_vector[1] ) ) /math.sqrt(4) )
    speed_maximizing_factor = (abs(velocity_vector[0]) + abs(velocity_vector[1])) / 4
    return 2 * translational_motion_factor + 2 * speed_maximizing_factor

def plot_figure(x, y, theta):
        """Plots the environment graphical interface
        along with the robot's current position"""
        ax = fig.add_subplot(111, aspect='equal')
        plt.xlim(-10, 110)
        plt.ylim(-10, 110)
        Rectangle = patches.Rectangle((0, 0), 100, 100, color='gray')
        ax.add_patch(Rectangle)

        left_bottom = patches.Circle((0, 100), radius=2, color='lime')
        ax.add_patch(left_bottom)
        left_up = patches.Circle((100, 0), radius=2, color='lime')
        ax.add_patch(left_up)
        right_bottom = patches.Circle((0, 0), radius=2, color='lime')
        ax.add_patch(right_bottom)
        right_up = patches.Circle((100, 100), radius=2, color='lime')
        ax.add_patch(right_up)

        ax.text(1, 1, 'Space where RobotX can move', color='white')

        plt.xlim(-10, 110)
        plt.ylim(-10, 110)

        obstacle_line1 = lines.Line2D([40, 60], [70, 80], color= 'white')
        ax.add_line(obstacle_line1)
        obstacle_line2 = lines.Line2D([40, 40], [40, 60], color = 'white')
        ax.add_line(obstacle_line2)

        line_left1 = patches.Circle((40, 70), radius=2, color='lime')
        ax.add_patch(line_left1)
        line_right1 = patches.Circle((60, 80), radius=2, color='lime')
        ax.add_patch(line_right1)
        line_left2 = patches.Circle((40, 40), radius=2, color='lime')
        ax.add_patch(line_left2)
        line_right2 = patches.Circle((40, 60), radius=2, color='lime')
        ax.add_patch(line_right2)


        ax.add_patch(patches.Circle((x, y), 2, color='yellow'))

        ax.add_patch(patches.Arrow(x, y,
                                   3 * math.cos(theta),
                                   3 * math.sin(theta),
                                   width=1, color='tomato'))
        plt.xlabel('X is in meters. X Position: {:0.2f}'.format(x))
        plt.ylabel('Y in meters. Y Position: {:0.2f}'.format(y))
        plt.title("Translational velocity: {:0.2f}".format(robo_x.translational_velocity) + ' meters/second')


if __name__ == "__main__":

    #------------------ Robot's Instantiation----------------------------#
    robo_x = rs.Robot()
    #--------------------------------------------------------------------#


    #------------------ Initialization of Robot's position----------------#
    robo_x.x = 30
    robo_x.y = 40
    robo_x.theta = math.pi/2
    #---------------------------------------------------------------------#


    #---------------------- Initial belief (mean and covariance) ------------------------#
    #   mean is set to the initial position
    #   covariance matrix is a zero matrix since initial position is known
    #   to the robot.
    mean_vector = np.array([robo_x.x, robo_x.y, robo_x.theta])
    mean_vector.shape = (3,1)
    covariance_matrix = np.matrix('0 0 0; 0 0 0; 0 0 0')
    #------------------------------------------------------------------------------------#

    #------------Instantiation ------------------------------------------#
    ev_alg = ea.Generation(POPULATION_SIZE,SELECTION_COUNT)
    nn = rs.Neural_Net()
    sensor = sm.Sensor_Model()
    #--------------------------------------------------------------------#

    robo_x.sense_environment(robo_x.x, robo_x.y)

    fig = plt.figure()
    plot_figure(robo_x.x, robo_x.y, robo_x.theta)
    plt.pause(1)

    #----------- Matrices used for Kalman filters -----------------------#
    np_A_matrix = np.matrix('1 0 0; 0 1 0; 0 0 1')
    np_A_matrix_transpose = np.transpose(np_A_matrix)

    np_C_matrix  = np.matrix('1 0 0; 0 1 0; 0 0 1')
    np_C_matrix_transpose = np.transpose(np_C_matrix)

    np_I_matrix = np_C_matrix

    np_R_matrix = np.matrix(' .5  0  0; 0  .5  0; 0  0  .09')
    np_Q_matrix = np.matrix(' 1.5, 0  0; 0  1.5  0; 0  0  .09')
    #-------------------------------------------------------------------#

    difference_array = []

    for i in range(300):

        for gene in ev_alg.genes:

            #-------------- dividing the individual among feedback and feedforward weights-----------#
            weightVector = gene.weightVector
            weight_matrix ,feedback_matrix = np.split(weightVector,[SPLIT_POINT])
            weight_matrix = np.reshape(weight_matrix,(nn.input_nodes + 1,nn.output_nodes))
            feedback_matrix = np.reshape(feedback_matrix,(nn.output_nodes,nn.output_nodes))
            #----------------------------------------------------------------------------------------#


            #--------- Controller based on Neural Network ------------------------------------------#
            nn.feedback_network(robo_x.np_sensor_output,weight_matrix,feedback_matrix)
            velocity_vector = nn.transform_velocity()
            #---------------------------------------------------------------------------------------#

            gene.fitness = calculateFitness(velocity_vector)

            #------------- Actual Moment including noise -------------------------------------------#
            B_x_U = robo_x.motion_model(velocity_vector[0], velocity_vector[1],
                                        np.reshape( np.array([robo_x.x, robo_x.y, robo_x.theta]), (3,1) )
                                        )

            actual_position = np.array([robo_x.x, robo_x.y, robo_x.theta]) \
                              + B_x_U + np.random.multivariate_normal([0, 0, 0], np_R_matrix)
            robo_x.update_position(actual_position[0], actual_position[1], actual_position[2])


            #------------- Calculating predicted belief from motion model --------------------------#
            B_x_U = robo_x.motion_model(velocity_vector[0], velocity_vector[1], mean_vector)
            B_x_U.shape = (3,1)
            predicted_mean_vector = np.dot(np_A_matrix, mean_vector) + B_x_U
            predicted_covariance_martrix = \
                np.dot(np.dot(np_A_matrix, covariance_matrix),np_A_matrix_transpose) + np_R_matrix
            #----------------------------------------------------------------------------------------#


            #------------- Calculating Kalmar Gain ---------------------------------------------------#
            np_K_matrix = \
            np.dot(
                np.dot(predicted_covariance_martrix, np_C_matrix_transpose),
                np.linalg.inv(np.dot(
                    np.dot(np_C_matrix, predicted_covariance_martrix),
                    np_C_matrix_transpose
                ) + np_Q_matrix)
            )
            #---------------------------------------------------------------------------------------#


            #--------- Calculating measurement from sensors (triangualation)#  ---------------------#
            C_x_X = sensor.scan_environment(tuple(( robo_x.x, robo_x.y )), robo_x.theta)
            print("Sensor_output:"+str(C_x_X));
            C_x_X.shape = (3,1)
            Z = C_x_X + np.reshape(np.random.multivariate_normal(np.zeros(3), np_Q_matrix ), (3,1))
            Z.shape = (3, 1)
            #---------------------------------------------------------------------------------------#


            #------  Calculating new belief --------------------------------------------------------#
            new_mean_vector = predicted_mean_vector + np.dot(
                np_K_matrix, Z - np.dot(np_C_matrix,predicted_mean_vector)
            )
            new_covariance_matrix = np.dot(
                np_I_matrix - np.dot(np_K_matrix, np_C_matrix),
                predicted_covariance_martrix
            )
            mean_vector = new_mean_vector
            mean_vector.shape = (3,1)
            covariance_matrix  = new_covariance_matrix
            #---------------------------------------------------------------------------------------#

            plot_figure(robo_x.x, robo_x.y, robo_x.theta)

            #----------------- Sampling from the belief for plotting ----------------------------------------------#
            samples = np.random.multivariate_normal( [ mean_vector[0,0],mean_vector[1,0], mean_vector[2,0] ],
                                                   covariance_matrix, 20)
            for sample in samples:
                ax = fig.add_subplot(111, aspect='equal')
                ax.add_patch(patches.Circle((sample[0], sample[1]), 2, color='linen', fill = False, linewidth= .5,
                                            ))
            #------------------------------------------------------------------------------------------------------#

            plt.pause(0.1)
            robo_x.collision_check()
            robo_x.sense_environment(robo_x.x, robo_x.y)
            plt.clf()

            #------------------- Calculating the difference of mean and actual position -------------------#
            difference_in_position = math.sqrt(np.sum(np.square(actual_position - mean_vector)))

        difference_array.append(difference_in_position)
        ev_alg.selection()
        ev_alg.reproduction()
        ev_alg.mutation(MUTATION_PROBABILITY)

    plt.plot(difference_array)
    plt.show()