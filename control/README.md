# control/

Lane-centering control module.

Takes detected lane geometry from `perception/`, computes the vehicle's lateral offset and
heading error relative to lane center, and converts these into steering commands using a
classical geometric or PID-based controller.
