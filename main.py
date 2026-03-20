from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import random

app = FastAPI()

# Định nghĩa cấu trúc dữ liệu đầu vào từ Google Sheets
class DataInput(BaseModel):
    feature1: float
    feature2: float
    feature3: float

# Định nghĩa cấu trúc dữ liệu đầu ra trả về kết quả dự đoán
class PredictionOutput(BaseModel):
    prediction: float
    status: str

@app.get("/")
def read_root():
    return {"message": "Dự đoán API đang hoạt động!"}

@app.post("/predict", response_model=PredictionOutput)
def predict(data: DataInput):
    try:
        # Giả lập mô hình dự đoán (Bạn có thể thay thế bằng mô hình thực tế của mình)
        # Ví dụ: Một phép toán đơn giản hoặc load model từ file .pkl
        prediction_result = (data.feature1 * 0.5) + (data.feature2 * 0.3) + (data.feature3 * 0.2)
        
        # Thêm một chút ngẫu nhiên để thấy sự thay đổi
        prediction_result += random.uniform(-0.1, 0.1)
        
        return PredictionOutput(prediction=round(prediction_result, 2), status="Success")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
