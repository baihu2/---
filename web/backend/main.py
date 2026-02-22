"""车牌识别系统后端服务 - 支持静态图片 + 实时视频 (FastAPI + YOLOv8 + CRNN)"""
import os
import io
import base64
import logging
from datetime import datetime
from typing import Optional, List, Dict
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import cv2
import numpy as np
import torch
from PIL import Image
import ultralytics
from ultralytics import YOLO
import uuid
import asyncio
# ====== 实时识别状态管理 ======
from collections import deque
import time

# 全局状态（简单场景可用，生产环境建议用 session 或 connection 级状态）
live_state = {
    "recent_results": deque(maxlen=5),  # 缓存最近5帧结果
    "last_valid_result": None,
    "last_seen_time": 0,
    "stable_plate": None,
    "stable_confidence": 0.0
}
# ======================
# 配置管理
# ======================
class Settings:
    YOLO_MODEL_PATH = r"C:\Users\白狐\Desktop\新建文件夹\BiYeSheJi\模型训练\runs\detect\plate_detection3\weights\best.pt"
    CRNN_MODEL_PATH = r"C:\BiYeSji\CRNN\crnn_best.pth"
    HOST = "0.0.0.0"
    PORT = 8000
    DEBUG = False
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
    DB_URL = "mysql+pymysql://user:password@localhost/license_plate_db"
    REDIS_URL = "redis://localhost:6379/0"

settings = Settings()

# ======================
# 字符集定义
# ======================
PROVINCES = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
             '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
             '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁', '新',
             '港', '澳', '挂', '学', '领', '使', '临']
LETTERS = [chr(ord('A') + i) for i in range(26)]
DIGITS = [str(i) for i in range(10)]
CHARS = ['<blank>'] + PROVINCES + LETTERS + DIGITS
IDX2CHAR = {idx: ch for idx, ch in enumerate(CHARS)}

# ======================
# Pydantic模型
# ======================
class RecognitionResult(BaseModel):
    success: bool
    plate_number: Optional[str] = None
    confidence: Optional[float] = None
    cropped_image: Optional[str] = None
    processing_time_ms: Optional[int] = None
    error_message: Optional[str] = None
    plate_type: Optional[str] = None
    timestamp: str = datetime.now().isoformat()

class PlateInfo(BaseModel):
    plate_number: str
    confidence: float
    bbox: List[int]
    plate_type: str
    timestamp: str

# ======================
# CRNN模型定义
# ======================
class CRNN(torch.nn.Module):
    def __init__(self, num_classes, imgH=32, nc=1, nh=256):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH must be a multiple of 16'
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(nc, 64, 3, 1, 1), torch.nn.ReLU(True), torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(64, 128, 3, 1, 1), torch.nn.ReLU(True), torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(128, 256, 3, 1, 1), torch.nn.BatchNorm2d(256), torch.nn.ReLU(True),
            torch.nn.Conv2d(256, 256, 3, 1, 1), torch.nn.ReLU(True), 
            torch.nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
            torch.nn.Conv2d(256, 512, 3, 1, 1), torch.nn.BatchNorm2d(512), torch.nn.ReLU(True),
            torch.nn.Conv2d(512, 512, 3, 1, 1), torch.nn.ReLU(True), 
            torch.nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
            torch.nn.Conv2d(512, 512, 2, 1, 0), torch.nn.ReLU(True)
        )
        self.rnn = torch.nn.LSTM(512, nh, num_layers=2, bidirectional=True, batch_first=True)
        self.fc = torch.nn.Linear(nh * 2, num_classes)
        
    def forward(self, x):
        conv = self.cnn(x)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        rnn_input = conv.squeeze(2).permute(0, 2, 1)
        rnn_out, _ = self.rnn(rnn_input)
        output = self.fc(rnn_out)
        return output

# ======================
# 辅助函数 (必须在模型加载前定义)
# ======================
def predict_plate_type(plate_img):
    if len(plate_img.shape) == 2:
        plate_img = cv2.cvtColor(plate_img, cv2.COLOR_GRAY2BGR)
    
    mean_color = np.mean(plate_img, axis=(0, 1))
    b, g, r = mean_color
    total = r + g + b
    if total == 0:
        return "unknown"
    
    r_ratio = r / total
    g_ratio = g / total
    b_ratio = b / total
    
    if g_ratio > 0.35 and g_ratio > r_ratio and g_ratio > b_ratio:
        return "green"
    elif b_ratio > 0.25 and b_ratio > r_ratio:
        return "blue"
    elif r_ratio > 0.4 and g_ratio > 0.3:
        return "yellow"
    else:
        return "unknown"

def format_plate_number(plate_str, plate_type):
    if not plate_str or len(plate_str) < 5:
        return plate_str
    
    if plate_type == "green":
        if len(plate_str) >= 8:
            return plate_str[:2] + "·" + plate_str[2:]
        elif len(plate_str) == 7:
            return plate_str[:2] + "·" + plate_str[2:]
        else:
            return plate_str
    
    return plate_str[:7] if len(plate_str) > 7 else plate_str

def preprocess_for_crnn(image, img_height=32, img_width=280):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    image = clahe.apply(image)
    
    image = cv2.resize(image, (img_width, img_height))
    image = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)
    image = image / 255.0
    image = (image - 0.5) / 0.5
    return image

def decode_ctc(outputs):
    _, preds = outputs.max(2)
    preds = preds.transpose(1, 0).cpu().numpy()
    decoded = []
    for seq in preds:
        out = []
        prev = -1
        for p in seq:
            if p != prev and p != 0:
                out.append(IDX2CHAR[p])
            prev = p
        plate_str = ''.join(out)
        decoded.append(plate_str)
    return decoded

# ======================
# 应用初始化 (只定义一次!)
# ======================
app = FastAPI(
    title="智能车牌识别系统API",
    description="支持静态图片上传 + 实时视频流识别",
    version="1.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 全局变量
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
yolo_model = None
crnn_model = None
task_queue: Dict[str, Dict] = {}

# ======================
# 核心识别函数
# ======================
async def recognize_plate_from_image(img: np.ndarray, filename: str = "") -> dict:
    try:
        results = yolo_model(img, verbose=False)
        if len(results[0].boxes) == 0:
            return {"plate_number": "", "confidence": 0.0, "error": "未检测到车牌"}

        boxes = results[0].boxes
        confidences = boxes.conf.cpu().numpy()
        best_idx = int(np.argmax(confidences))
        box = boxes.xyxy[best_idx].cpu().numpy().astype(int)
        confidence = float(confidences[best_idx])
        x1, y1, x2, y2 = box

        pad = max(5, int(min(x2-x1, y2-y1) * 0.1))
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(img.shape[1], x2 + pad)
        y2 = min(img.shape[0], y2 + pad)
        plate_img = img[y1:y2, x1:x2]

        crnn_input = preprocess_for_crnn(plate_img).to(device)
        with torch.no_grad():
            crnn_output = crnn_model(crnn_input)
        plate_number_raw = decode_ctc(crnn_output.permute(1, 0, 2))[0]

        if not plate_number_raw or len(plate_number_raw) < 5:
            return {"plate_number": "", "confidence": confidence, "error": "字符识别失败"}

        plate_type = predict_plate_type(plate_img)
        plate_number = format_plate_number(plate_number_raw, plate_type)

        cropped_base64 = None
        if filename:
            _, buffer = cv2.imencode('.jpg', plate_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            b64 = base64.b64encode(buffer).decode('utf-8')
            cropped_base64 = f"data:image/jpeg;base64,{b64}"

        return {
            "plate_number": plate_number,
            "confidence": round(confidence, 4),
            "plate_type": plate_type,
            "cropped_image": cropped_base64,
            "file_name": filename
        }
    except Exception as e:
        return {"plate_number": "", "confidence": 0.0, "error": str(e)}

# ======================
# 生命周期事件
# ======================
@app.on_event("startup")
async def load_models():
    global yolo_model, crnn_model
    logger.info(f"使用设备: {device}")
    
    try:
        logger.info("正在加载YOLO车牌检测模型...")
        yolo_model = YOLO(settings.YOLO_MODEL_PATH)
        yolo_model.to(device)
        logger.info("✅ YOLO模型加载成功")
    except Exception as e:
        logger.error(f"❌ YOLO模型加载失败: {e}")
        raise
    
    try:
        logger.info("正在加载CRNN字符识别模型...")
        crnn_model = CRNN(num_classes=len(CHARS)).to(device)
        crnn_model.load_state_dict(
            torch.load(settings.CRNN_MODEL_PATH, map_location=device)
        )
        crnn_model.eval()
        logger.info("✅ CRNN模型加载成功")
    except Exception as e:
        logger.error(f"❌ CRNN模型加载失败: {e}")
        raise
    
    logger.info("正在预热模型...")
    dummy_img = np.zeros((32, 280, 3), dtype=np.uint8)
    _ = preprocess_for_crnn(dummy_img)
    logger.info("✅ 模型预热完成")

@app.on_event("shutdown")
async def cleanup():
    global yolo_model, crnn_model
    del yolo_model
    del crnn_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("✅ 资源清理完成")

# ======================
# API端点
# ======================
@app.get("/healthz")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": yolo_model is not None and crnn_model is not None,
        "device": str(device)
    }

@app.post("/api/v1/recognize", response_model=RecognitionResult)
async def recognize_plate(file: UploadFile = File(...)):
    start_time = datetime.now()
    
    # 1. 验证文件
    try:
        content = await file.read()
        if len(content) > settings.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413, 
                detail=f"文件过大，最大支持{settings.MAX_FILE_SIZE/1024/1024}MB"
            )
        
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in settings.ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=415,
                detail=f"不支持的文件类型，仅支持{', '.join(settings.ALLOWED_EXTENSIONS)}"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"文件处理失败: {e}")
        return RecognitionResult(
            success=False,
            error_message=f"文件处理失败: {str(e)}",
            timestamp=datetime.now().isoformat()
        )
    
    # 2. 识别处理
    try:
        nparr = np.frombuffer(content, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("无法解析图像文件")
            
        result = await recognize_plate_from_image(img, file.filename)
        
        if "error" in result:
            return RecognitionResult(
                success=False,
                error_message=result["error"],
                timestamp=datetime.now().isoformat()
            )
        
        processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
        
        return RecognitionResult(
            success=True,
            plate_number=result["plate_number"],
            confidence=result["confidence"],
            cropped_image=result["cropped_image"],
            processing_time_ms=processing_time,
            plate_type=result["plate_type"],
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"识别过程异常: {e}")
        return RecognitionResult(
            success=False,
            error_message=f"识别过程异常: {str(e)}",
            timestamp=datetime.now().isoformat()
        )

# ======================
# WebSocket 实时识别
# ======================
@app.websocket("/ws/live")
@app.websocket("/ws/live")
async def live_recognition(websocket: WebSocket):
    await websocket.accept()
    logger.info("🟢 实时识别 WebSocket 连接建立")

    # 为每个连接创建独立状态
    state = {
        "recent_results": deque(maxlen=5),      # 用于稳定性投票
        "last_seen_time": time.time(),
        "best_plate": None,                     # 最佳车牌号
        "best_confidence": 0.0,                 # 最佳置信度
        "best_cropped_image": "",               # 最佳裁剪图（Base64）
        "stable_plate": None,                   # 当前稳定输出的车牌
    }

    try:
        while True:
            data = await websocket.receive_bytes()
            nparr = np.frombuffer(data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                continue

            # 使用核心识别函数
            result = await recognize_plate_from_image(img)
            current_time = time.time()

            plate_number = result.get("plate_number")
            confidence = result.get("confidence", 0.0)
            cropped_image = result.get("cropped_image", "")

            # === 更新“最佳车牌”缓存 ===
            if plate_number and confidence > state["best_confidence"]:
                state["best_plate"] = plate_number
                state["best_confidence"] = confidence
                state["best_cropped_image"] = cropped_image
                logger.debug(f"🏆 更新最佳车牌: {plate_number} (置信度: {confidence:.2f})")

            # === 稳定性投票逻辑（同之前）===
            if plate_number:
                state["recent_results"].append({
                    "plate": plate_number,
                    "confidence": confidence,
                    "timestamp": current_time
                })
                state["last_seen_time"] = current_time
            else:
                # 超时清空 recent_results（但保留 best）
                if current_time - state["last_seen_time"] > 3.0:
                    state["recent_results"].clear()

            # 投票决定当前稳定车牌
            plate_votes = {}
            for r in state["recent_results"]:
                plate = r["plate"]
                plate_votes[plate] = plate_votes.get(plate, 0) + 1

            stable_plate = None
            for plate, votes in plate_votes.items():
                if votes >= 2:
                    stable_plate = plate
                    break

            state["stable_plate"] = stable_plate

            # === 构建响应：优先使用“最佳”裁剪图 ===
            final_plate = state["best_plate"] if state["best_plate"] else ""
            final_confidence = state["best_confidence"]
            final_cropped_image = state["best_cropped_image"]

            # 如果长时间未见车牌（比如5秒），清空最佳结果
            if current_time - state["last_seen_time"] > 5.0:
                final_plate = ""
                final_cropped_image = ""
                final_confidence = 0.0

            response = {
                "success": bool(final_plate),
                "plate_number": final_plate,
                "plate_type": result.get("plate_type", "unknown"),
                "confidence": final_confidence,
                "cropped_image": final_cropped_image,
                "timestamp": datetime.now().isoformat()
            }

            await websocket.send_json(response)

    except Exception as e:
        logger.error(f"🔴 WebSocket 错误: {e}")
    finally:
        await websocket.close()
        logger.info("🔴 实时识别 WebSocket 连接关闭")
# ======================
# 批量处理 (保持原有功能)
# ======================
async def process_batch_task(task_id: str, files_data: List[dict]):
    try:
        task = task_queue.get(task_id)
        if not task:
            return
        
        results = []
        for file_data in files_data:
            nparr = np.frombuffer(file_data["content"], np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            result = await recognize_plate_from_image(img, file_data["filename"])
            results.append(result)
            task["processed_count"] += 1
            task["results"].append(result)
            await asyncio.sleep(0.1)
        
        task["status"] = "completed"
        task["end_time"] = datetime.now().isoformat()
        logger.info(f"批量任务 {task_id} 完成")
        
    except Exception as e:
        logger.error(f"批量任务 {task_id} 处理失败: {e}")
        if task_id in task_queue:
            task_queue[task_id]["status"] = "failed"
            task_queue[task_id]["error"] = str(e)

@app.post("/api/v1/recognize-batch")
async def recognize_batch(files: List[UploadFile] = File(...), background_tasks: BackgroundTasks = None):
    # ... [您的原有批量处理逻辑，调用 recognize_plate_from_image] ...
    try:
        if not files:
            raise HTTPException(status_code=400, detail="至少需要上传一个文件")
        
        files_data = []
        for file in files:
            content = await file.read()
            if len(content) > settings.MAX_FILE_SIZE:
                raise HTTPException(status_code=413, detail=f"文件 {file.filename} 过大")
            ext = os.path.splitext(file.filename)[1].lower()
            if ext not in settings.ALLOWED_EXTENSIONS:
                raise HTTPException(status_code=415, detail=f"文件 {file.filename} 类型不支持")
            files_data.append({"content": content, "filename": file.filename})
        
        task_id = str(uuid.uuid4())
        task_queue[task_id] = {
            "status": "processing",
            "files_count": len(files_data),
            "processed_count": 0,
            "results": [],
            "start_time": datetime.now().isoformat()
        }
        
        background_tasks.add_task(process_batch_task, task_id, files_data)
        return {
            "success": True,
            "task_id": task_id,
            "message": f"批量识别任务已创建，共{len(files_data)}张图片",
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"批量识别失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/task/{task_id}")
async def get_task_status(task_id: str):
    task = task_queue.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    return {
        "task_id": task_id,
        "status": task["status"],
        "files_count": task["files_count"],
        "processed_count": task["processed_count"],
        "results": task["results"] if task["status"] == "completed" else [],
        "error": task.get("error"),
        "start_time": task["start_time"],
        "end_time": task.get("end_time")
    }

# ======================
# 启动应用
# ======================
if __name__ == "__main__":
    import uvicorn
    logger.info(f"🚀 启动车牌识别API服务 (端口: {settings.PORT})")
    logger.info(f"📄 API文档: http://localhost:{settings.PORT}/docs")
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=1 if torch.cuda.is_available() else 4
    )