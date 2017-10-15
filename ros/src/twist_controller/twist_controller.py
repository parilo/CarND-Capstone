from yaw_controller import YawController
from pid import PID
import rospy
from lowpass import LowPassFilter
import math

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self,
        	wheel_base,
        	steer_ratio,
        	min_speed,
        	max_lat_accel,
        	max_steer_angle,
            decel_limit,
            accel_limit):
        # TODO: Implement
        self.YawController = YawController(
        	wheel_base,
        	steer_ratio,
        	min_speed,
        	max_lat_accel,
        	max_steer_angle)

        self.PID = PID(0.9, 0.0005, 0.075, decel_limit, accel_limit)
        # self.PID = PID(4.0, 0.001, 0.05, decel_limit, accel_limit)
        self.low_pass_filer_vel = LowPassFilter(10.0, 1.0)

        self.lastT = None
        self.last_dbw_enabled = False


    def control(self, linear_velocity, angular_velocity, current_velocity, dbw_enabled):
        if dbw_enabled is True and self.last_dbw_enabled is False:
    		#restart
    		self.lastT = None
    		self.PID.reset()
    	self.last_dbw_enabled = dbw_enabled

        self.low_pass_filer_vel.filt(linear_velocity)
        linear_velocity = self.low_pass_filer_vel.get()

    	velocity_error = linear_velocity - current_velocity
    	T = rospy.get_time()
    	dt = T - self.lastT if self.lastT else 0.05
    	self.lastT = T
    	a = self.PID.step(velocity_error,dt)
     	if a > 0.0:
    		throttle, brake = a, 0.0
    	else:
    		throttle, brake = 0.0, math.fabs(a)

        steer = self.YawController.get_steering(linear_velocity, angular_velocity, current_velocity)

        # print('--- pid: {} {}'.format(velocity_error, a))

        return throttle, brake, steer
