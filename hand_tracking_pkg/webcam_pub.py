import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

def publish_image():
   image_pub = rospy.Publisher("image_raw", Image, queue_size=10)
   bridge = CvBridge()
   capture = cv2.VideoCapture("/dev/video0")

   while not rospy.is_shutdown():
    ret, img = capture.read()
    if not ret:
        rospy.ERROR("could not grad a frame!")
        break
    try:
        img_msg = bridge.cv2_to_imgmsg(img,"bgr8")
        image_pub.publish(img_msg)
    except CvBridgeError as error:
        print(error)

if __name__ == "main":
    rospy.init_node("my_cam", anonymous = True)
    print("Image is being published to the topci /image_raw ...")
    publish_image()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("shutting down!")
