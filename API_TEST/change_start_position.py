import airsim

client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(False)

# position = Vector3r(x,y,z)
position = airsim.Vector3r(39, 94, 9.9)
# heading = AirSimClientBase.toQuaternion(roll, pitch, yaw)
heading = airsim.utils.to_quaternion(0, 0, 2.5)
pose = airsim.Pose(position, heading)
client.simSetVehiclePose(pose, True)
