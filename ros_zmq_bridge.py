import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import zmq

class ZmqVisionBridge(Node):
    def __init__(self):
        super().__init__('zmq_vision_bridge')
        self.bridge = CvBridge()
        
        # 🚀 修正點：將變數名稱改為 zmq_context 與 zmq_socket，避免與 ROS Node 衝突
        self.zmq_context = zmq.Context()
        self.zmq_socket = self.zmq_context.socket(zmq.REQ)
        self.zmq_socket.connect("tcp://127.0.0.1:5555")
        
        # 訂閱相機 Topic
        self.subscription = self.create_subscription(
            Image,
            '/side_cam/rgb',
            self.image_callback,
            10
        )
        
        # 發佈 Mask 與除錯畫面
        self.mask_pub = self.create_publisher(Image, '/side_cam/mask', 10)
        self.overlay_pub = self.create_publisher(Image, '/side_cam/tracking_overlay', 10)
        
        self.get_logger().info("✅ ROS <-> ZMQ 橋接節點已啟動！")

    def image_callback(self, msg):
        # 1. 將 ROS 影像轉為 OpenCV 格式
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        height, width, _ = cv_image.shape
        
        # 2. 編碼成 JPG 再傳輸
        _, img_encoded = cv2.imencode('.jpg', cv_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        
        # 🚀 修正點：使用 zmq_socket 發送與接收
        self.zmq_socket.send(img_encoded.tobytes())
        
        # 3. 阻塞等待 AI Server 回傳 Mask
        mask_bytes = self.zmq_socket.recv()
        
        # 4. 重建 Mask 陣列
        mask_uint8 = np.frombuffer(mask_bytes, dtype=np.uint8).reshape(height, width)
        
        # --- 下方發佈 ROS 訊息的程式碼維持不變 ---
        mask_255 = (mask_uint8 * 255).astype(np.uint8)
        mask_msg = self.bridge.cv2_to_imgmsg(mask_255, encoding="mono8")
        mask_msg.header = msg.header 
        self.mask_pub.publish(mask_msg)
        
        color_mask = np.zeros_like(cv_image)
        color_mask[mask_uint8 == 1] = [0, 255, 0]
        overlay_frame = cv2.addWeighted(cv_image, 1.0, color_mask, 0.5, 0)
        
        overlay_msg = self.bridge.cv2_to_imgmsg(overlay_frame, encoding="bgr8")
        overlay_msg.header = msg.header
        self.overlay_pub.publish(overlay_msg)

def main(args=None):
    rclpy.init(args=args)
    node = ZmqVisionBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
