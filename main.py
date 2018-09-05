#!/usr/bin/python3
# top level script to run the show

# TODO: add all the ncs stuff
# TODO: add all the gimbal control stuff
# TODO: add the EKF
# TODO: add the control stuff


init_ncs()
init_gimbal()

while True:
    pic = take_picture()

	obj_position = run_through_ncs(pic)
	
	# while the nn recognises the image (approx 80ms),
	# do other stuff:
	
	angles = get_angles_from_gimbal()

	states = EKF(obj_position, angles)
	
	control_action = controller(states)
	
	command_gimbal(control_action)
