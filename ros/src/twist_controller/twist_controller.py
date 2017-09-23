from yaw_controller import YawController
from pid import PID
import rospy

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
        
        self.lastT = None
        self.last_dbw_enabled = False


    def control(self, linear_velocity, angular_velocity, current_velocity, dbw_enabled):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer
        if dbw_enabled is True and self.last_dbw_enabled is False:
    		#restart
    		self.lastT = None
    		self.PID.reset()
    	self.last_dbw_enabled = dbw_enabled

    	velocity_error = linear_velocity - current_velocity
    	T = rospy.get_time()
    	dt = T - self.lastT if self.lastT else 0.05
    	self.lastT = T
    	a = self.PID.step(velocity_error,dt)
     	if a > 0.0:
    		throttle, brake = a, 0.0
    	else:
    		throttle, brake = 0.0, abs(a)

        steer = self.YawController.get_steering(linear_velocity, angular_velocity, current_velocity)
        
        return throttle, brake, steer
