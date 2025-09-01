from ultralytics import YOLO
import cv2
import paho.mqtt.client as mqtt
import json
import base64
import time


class SimpleHelmetDetector:
    def __init__(self, model_path='helmet.pt', mqtt_broker='localhost', mqtt_port=1883):
        """
        极简安全帽检测器
        :param model_path: 模型路径
        :param mqtt_broker: MQTT代理地址
        :param mqtt_port: MQTT代理端口
        """
        self.model = YOLO(model_path)
        self.mqtt_client = mqtt.Client()
        self.mqtt_client.connect(mqtt_broker, mqtt_port)
        self.mqtt_topic = "alert"
        self.target_classes_id = [1]  # 安全帽的类别ID

    def detect(self, frame):
        """
        执行安全帽检测
        :param frame: 输入图像帧
        :return: 检测结果列表，每个元素包含类别和置信度
        """
        results = self.model(frame)
        detections = []

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls)
                # 只收集目标类别的检测结果
                if class_id == self.target_classes_id:
                    detections.append({
                        'class_id': class_id,
                        'confidence': float(box.conf)
                    })

        # 发送检测结果
        if detections:
            self._send_detection_result(frame, detections)
        return detections

    def _send_detection_result(self, frame, detections):
        """通过MQTT发送检测结果"""
        _, buffer = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')

        message = json.dumps({
            "timestamp": int(time.time()),
            "detections": detections,
            "type": 79,
            "typeName": "人员未带安全帽",
            "frame_base64": frame_base64
        }, ensure_ascii=False)

        self.mqtt_client.publish(self.mqtt_topic, message)

    def process_video(self, video_path, frame_skip=5):
        """
        处理视频文件
        :param video_path: 视频文件路径
        :param frame_skip: 跳帧数，提高处理速度
        """
        cap = cv2.VideoCapture(video_path)
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_skip == 0:
                self.detect(frame)

            frame_count += 1

        cap.release()
        self.mqtt_client.disconnect()


# 使用示例
if __name__ == "__main__":
    detector = SimpleHelmetDetector(model_path='./helmet.pt')
    detector.process_video("./test.mp4")
