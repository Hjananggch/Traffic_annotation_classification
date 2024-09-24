import gradio as gr
import onnxruntime as ort
from PIL import Image
from torchvision import transforms
import numpy as np

ort_session = ort.InferenceSession(r'./save_model/model.onnx')

class_names = {
    0: "限速5km", 1: "限速15km", 2: "限速30km", 3: "限速40km", 5: "限速60km",
    6: "限速70km", 7: "限速80km", 8: "禁止左转和直行", 9: "禁止直行和右转",
    10: "禁止直行", 11: "禁止左转", 12: "禁止左右转弯", 14: "禁止超车",
    15: "禁止掉头", 16: "禁止机动车驶入", 17: "禁止鸣笛", 18: "解除40km限制",
    19: "解除50km限制", 20: "直行和右转", 21: "单直行", 22: "向左转弯",
    23: "向左向右转弯", 24: "向右转弯", 25: "靠左侧通道行驶", 26: "靠右侧道路行驶",
    27: "环岛行驶", 28: "机动车行驶", 29: "鸣喇叭", 30: "非机动车行驶",
    31: "允许掉头", 32: "左右绕行", 33: "注意红绿灯", 34: "注意危险",
    35: "注意行人", 36: "注意非机动车", 37: "注意儿童", 38: "向右急转弯",
    39: "向左急转弯", 40: "下陡坡", 41: "上陡坡", 42: "慢行", 43: "T形交叉",
    44: "T形交叉", 45: "村庄", 46: "反向弯路", 47: "无人看守铁路道口",
    48: "施工", 49: "连续弯路", 50: "有人看守铁路道口", 51: "事故易发生路段",
    52: "停车让行", 53: "禁止通行", 54: "禁止车辆临时或长时间停放", 55: "禁止输入",
    56: "减速让行", 57: "停车检查"
}

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

def preprocess_image(image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).numpy()  # 这里 image 应该是 PIL 图像

# 模型推理函数
def classify_image(image):
    # 预处理图像
    image_np = preprocess_image(image)

    # 执行推理
    inputs = {ort_session.get_inputs()[0].name: image_np}
    outputs = ort_session.run(None, inputs)
    outputs_softmax = softmax(outputs[0])  # Softmax
    probabilities = np.max(outputs_softmax, axis=1)  # 计算最大概率值
    predicted_idx = np.argmax(outputs[0], axis=1)  # 获取预测的类别索

    # 设置置信度阈值
    confidence_threshold = 0.75
    # print(probabilities)
    # 检查置信度是否达标
    if probabilities[0] < confidence_threshold:
        return "置信度过低，无法分类"
    else:
        return class_names.get(predicted_idx[0], "类别未知")

examples = [
    r"./test_img\000_1_0002.png",
    r"./test_img\003_xs40.png"
]

iface = gr.Interface(
    fn=classify_image,  # 推理函数
    inputs=gr.Image(),  # 输入类型
    outputs=gr.Text(),  # 输出类型
    title="交通标志图像分类",  # 界面标题
    description="上传一张图片进行分类。模型能够识别不同类型的交通标志。",  # 界面描述
    examples=examples,  # 示例图片
    theme="huggingface",  
    css=".gradio-app {font-family: Arial;}"  
)

iface.launch(server_port=10010)


