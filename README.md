# Fruit and Vegetable Classification Flask App

<div align="center">
  <img src="https://img.shields.io/badge/Machine_Learning-Classification-blue?style=for-the-badge&logo=python">
  <img src="https://img.shields.io/badge/Framework-Flask-green?style=for-the-badge&logo=flask">
  <img src="https://img.shields.io/badge/Deep_Learning-CNN-orange?style=for-the-badge&logo=tensorflow">
  <img src="https://img.shields.io/badge/Deployment-Web_App-red?style=for-the-badge">
</div>

## 📋 Tổng Quan (Overview)

Ứng dụng web phân loại trái cây và rau củ sử dụng Deep Learning và Flask framework. Hệ thống có thể nhận diện 36 loại trái cây và rau củ khác nhau thông qua việc upload ảnh, chụp từ camera, hoặc nhập URL link ảnh. Đây là bài tập thực hành deploy Machine Learning model lên web.

This is a web application for fruit and vegetable classification using Deep Learning and Flask framework. The system can recognize 36 different types of fruits and vegetables through image upload, camera capture, or URL image input. This is a practical exercise for deploying Machine Learning models to the web.

## 🎯 Tính Năng Chính (Key Features)

- **36 Loại Thực Phẩm**: Nhận diện apple, banana, beetroot, bell pepper, cabbage, và nhiều loại khác
- **3 Cách Input**: Upload file, chụp camera real-time, hoặc nhập URL ảnh
- **Real-time Camera**: Sử dụng webcam để chụp và phân loại ngay lập tức
- **Top 3 Predictions**: Hiển thị 3 kết quả có xác suất cao nhất
- **Web Interface**: Giao diện thân thiện với Bootstrap và CSS
- **Easy Deployment**: Triển khai đơn giản với Flask framework

## 🍎 Danh Sách Các Loại (Supported Classes)

Hệ thống có thể nhận diện **36 loại** trái cây và rau củ:

### 🍓 Trái Cây (Fruits)
- **Apple** (Táo)
- **Banana** (Chuối) 
- **Grapes** (Nho)
- **Kiwi** (Kiwi)
- **Lemon** (Chanh)
- **Mango** (Xoài)
- **Orange** (Cam)
- **Pear** (Lê)
- **Pineapple** (Dứa)
- **Pomegranate** (Lựu)
- **Watermelon** (Dưa hấu)

### 🥕 Rau Củ (Vegetables)
- **Beetroot** (Củ dền)
- **Bell Pepper** (Ớt chuông)
- **Cabbage** (Bắp cải)
- **Capsicum** (Ớt ngọt)
- **Carrot** (Cà rót)
- **Cauliflower** (Súp lơ)
- **Chilli Pepper** (Ớt cay)
- **Corn** (Bắp)
- **Cucumber** (Dưa chuột)
- **Eggplant** (Cà tím)
- **Garlic** (Tỏi)
- **Ginger** (Gừng)
- **Jalepeno** (Ớt jalapeño)
- **Lettuce** (Xà lách)
- **Onion** (Hành tây)
- **Paprika** (Ớt bột)
- **Peas** (Đậu Hà Lan)
- **Potato** (Khoai tây)
- **Raddish** (Củ cải)
- **Soy Beans** (Đậu nành)
- **Spinach** (Rau bina)
- **Sweetcorn** (Bắp ngọt)
- **Sweetpotato** (Khoai lang)
- **Tomato** (Cà chua)
- **Turnip** (Củ cải trắng)

## 🏗️ Kiến Trúc Hệ Thống (System Architecture)

### 1. Machine Learning Model
- **Model Type**: Convolutional Neural Network (CNN)
- **Input Size**: 100x100x3 (RGB images)
- **Output**: 36 classes probability distribution
- **Model File**: `Fruit_and_vegetable_classification.h5`
- **Framework**: TensorFlow/Keras

### 2. Web Framework
- **Backend**: Flask (Python)
- **Frontend**: HTML, CSS, Bootstrap
- **Real-time**: OpenCV for camera integration
- **File Handling**: PIL for image processing

### 3. Input Methods
```python
Input Options:
1. File Upload → Image Processing → Prediction
2. Camera Capture → Real-time Frame → Prediction  
3. URL Input → Download Image → Prediction
```

## 🛠️ Công Nghệ Sử Dụng (Technology Stack)

### Backend Technologies
- **Flask**: Web framework
- **TensorFlow/Keras**: Deep learning model
- **OpenCV**: Camera and image processing
- **PIL/Pillow**: Image manipulation
- **NumPy**: Numerical computations

### Frontend Technologies
- **HTML5**: Web structure
- **CSS3**: Styling and layout
- **Bootstrap**: Responsive design
- **JavaScript/jQuery**: Interactive elements

### Dependencies
```python
flask
tensorflow
opencv-python
pillow
numpy
uuid
urllib
```

## ⚙️ Cài Đặt và Sử Dụng (Installation & Usage)

### 1. Clone Repository

```bash
git clone https://github.com/kenzn2/Fruit_and_vegetable_classification_flask.git
cd Fruit_and_vegetable_classification_flask
```

### 2. Install Dependencies

```bash
pip install flask tensorflow opencv-python pillow numpy
```

### 3. Project Structure

```
Fruit_and_vegetable_classification_flask/
├── app.py                                    # Main Flask application
├── Fruit_and_vegetable_classification.h5     # Trained CNN model
├── README.md                                 # Project documentation
├── static/
│   ├── css/                                 # CSS stylesheets
│   └── images/                              # Uploaded/captured images
└── templates/
    ├── index.html                           # Main page
    ├── success.html                         # Results page
    └── camera.html                          # Camera interface
```

### 4. Run Application

```bash
python app.py
```

Ứng dụng sẽ chạy tại: `http://localhost:5000`

### 5. Usage Instructions

#### 📁 File Upload Method
1. Truy cập trang chủ
2. Click "Choose File" và chọn ảnh trái cây/rau củ
3. Click "Predict" để nhận kết quả
4. Xem top 3 dự đoán với xác suất

#### 📷 Camera Method  
1. Click "Open Camera"
2. Cho phép truy cập camera
3. Đặt trái cây/rau củ trước camera
4. Click "Take Picture" để chụp và phân loại

#### 🔗 URL Method
1. Nhập URL của ảnh trái cây/rau củ
2. Click "Predict from URL"
3. Hệ thống sẽ tải và phân loại ảnh

## 📊 Kết Quả và Đánh Giá (Results & Performance)

### Model Performance
- **Input Resolution**: 100x100 pixels
- **Color Channels**: RGB (3 channels)
- **Batch Processing**: 32 images per batch
- **Output Format**: Top 3 predictions with probabilities

### Prediction Output Format
```python
Example Result:
{
    "class1": "apple",      "prob1": 85.67,
    "class2": "pear",       "prob2": 12.45,
    "class3": "orange",     "prob3": 1.88
}
```

### Supported Image Formats
- **JPEG** (.jpg, .jpeg)
- **PNG** (.png)
- **File Size**: Recommended < 5MB
- **Resolution**: Any (auto-resized to 100x100)

## 🔧 Chi Tiết Kỹ Thuật (Technical Details)

### Image Preprocessing Pipeline
```python
def predict(filename, model):
    # Load and resize image to 100x100
    img = image.load_img(filename, target_size=(100,100,3))
    
    # Convert to array and expand dimensions
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    
    # Model prediction
    result = model.predict(images, batch_size=32)
    
    # Process results to get top 3
    return top_3_classes, top_3_probabilities
```

### Flask Routes
```python
Routes:
├── '/'              → Home page (index.html)
├── '/success'       → Prediction results (success.html)
├── '/camera'        → Camera interface (camera.html)
├── '/takeimage'     → Camera capture processing
└── '/video_feed'    → Real-time video streaming
```

### Camera Integration
```python
# Real-time camera capture
video = cv2.VideoCapture(0)

def gen():
    while True:
        rval, frame = video.read()
        cv2.imwrite('t.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + 
               open('t.jpg', 'rb').read() + b'\r\n')
```

## 🎨 Giao Diện Người Dùng (User Interface)

### Main Features
- **Responsive Design**: Tương thích với mobile và desktop
- **Bootstrap Integration**: Modern và professional UI
- **Real-time Feedback**: Hiển thị kết quả ngay lập tức
- **Error Handling**: Thông báo lỗi khi input không hợp lệ

### Page Components
- **Header**: Tiêu đề và navigation
- **Upload Section**: File input và camera buttons
- **Results Display**: Top 3 predictions với percentages
- **Footer**: Thông tin team members

## 👥 Thành Viên Nhóm (Team Members)

Dự án được thực hiện bởi nhóm sinh viên:

- **Hồ Yến Ngọc**
- **Vũ Ngọc Tuấn** 
- **Nguyễn Anh Tuấn**
- **Lê Đình Chương**

## 🚀 Deployment Options

### Local Development
```bash
python app.py
# Access at http://localhost:5000
```

### Production Deployment
```bash
# Using Gunicorn
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app

# Using Docker
docker build -t fruit-classifier .
docker run -p 5000:5000 fruit-classifier
```

### Cloud Deployment
- **Heroku**: `git push heroku main`
- **AWS EC2**: Upload và run with gunicorn
- **Google Cloud**: Deploy với App Engine
- **Azure**: Web App deployment

## 🔧 Tối Ưu Hóa và Cải Tiến (Optimization & Improvements)

### Current Features
- ✅ 36 class classification
- ✅ Multiple input methods
- ✅ Real-time camera integration
- ✅ Web-based interface
- ✅ Top 3 predictions display

### Future Enhancements
- 🔄 **Model Improvements**: Higher accuracy with more data
- 📱 **Mobile App**: React Native or Flutter version
- 🎯 **More Classes**: Expand to 100+ food items
- 🔍 **Nutrition Info**: Add nutritional information display
- 📊 **Analytics Dashboard**: Usage statistics and insights
- 🌐 **Multi-language**: Support Vietnamese and English
- ⚡ **Performance**: Model optimization for faster inference

## 📚 Tài Liệu Tham Khảo (References)

- [Flask Documentation](https://flask.palletsprojects.com/) - Web framework guide
- [TensorFlow Keras](https://www.tensorflow.org/guide/keras) - Deep learning framework
- [OpenCV Python](https://opencv-python-tutroals.readthedocs.io/) - Computer vision library
- [Bootstrap](https://getbootstrap.com/) - Frontend framework
- [CNN for Image Classification](https://cs231n.github.io/convolutional-networks/) - Stanford CS231n
- [Deploy ML Models](https://towardsdatascience.com/deploy-machine-learning-model-on-google-kubernetes-engine-94daac85108b) - Deployment guide

## 🤝 Đóng Góp (Contributing)

1. Fork repository
2. Tạo feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Tạo Pull Request

### Development Guidelines
- Follow PEP 8 for Python code
- Test new features thoroughly
- Update documentation for changes
- Ensure responsive design for UI changes

## 📄 License

Dự án này được phân phối dưới MIT License. Xem file `LICENSE` để biết thêm chi tiết.

## 📞 Liên Hệ (Contact)

- **Repository**: [Fruit_and_vegetable_classification_flask](https://github.com/kenzn2/Fruit_and_vegetable_classification_flask)
- **Issues**: Báo cáo lỗi tại GitHub Issues
- **Discussions**: GitHub Discussions cho Q&A

## 🙏 Acknowledgments

- TensorFlow team for the deep learning framework
- Flask community for the web framework
- OpenCV developers for computer vision tools
- Bootstrap team for the UI framework
- Dataset contributors for training data

## 📱 Screenshots

### Main Interface
```
[Upload Section]
- File upload button
- Camera access button  
- URL input field

[Results Display]
- Uploaded/captured image
- Top 3 predictions with percentages
- Confidence scores
```

### Camera Interface
```
[Real-time Video Feed]
- Live camera stream
- Capture button
- Automatic classification
```

---

<div align="center">
  <b>🍎 Nhận diện trái cây và rau củ với AI! 🥕</b><br>
  <i>Deploy Machine Learning Model to Web with Flask</i>
</div>