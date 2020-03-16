import airsim

client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(False)