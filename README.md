# Transformers are Graph Neural Networks: Lý thuyết và Thực nghiệm Graph Transformer Networks trên Đồ thị Dị thể

Mã nguồn thực nghiệm, vá lỗi hệ thống và phân tích hiệu năng dựa trên bài toán Phân loại nút (Node Classification) nhằm kiểm chứng sự hội tụ toán học giữa cơ chế **Self-Attention (Transformer)** và **Message Passing (Mạng Nơ-ron Đồ thị - GNN)** trên đồ thị đầy đủ.

* **Nhóm học viên thực hiện:** 
  * Nguyễn Thị Ngọc Mẫn (MSHV: 2582003124)
  * Nguyễn Thị Thanh Mai (MSHV: 2582003044)
  * Đỗ Thị Lan (MSHV: 2582003122)
* **Môn học:** Phân tích kiến trúc học máy nâng cao
* **Đơn vị:** Phân hiệu Trường Đại học Thủy Lợi — TP. Hồ Chí Minh

---

## 📌 1. Khái quát Lý thuyết & Động lực Nghiên cứu
Nghiên cứu của Chaitanya K. Joshi đã chứng minh một sự hội tụ toán học sâu sắc: **Transformer thực chất là một dạng Đa tạp của Mạng Nơ-ron Đồ thị (GNN) hoạt động trên đồ thị đầy đủ (Fully Connected Graph)**, nơi các token trong câu đóng vai trò là các nút (nodes) và trọng số chú ý (Attention weights) đóng vai trò là các cạnh động (dynamic edges).

Dự án này triển khai kiến trúc mở rộng **Graph Transformer Networks (GTN / FastGTN)** nhằm:
* Vượt qua hai nhược điểm cốt hữu của GNN truyền thống là *Over-smoothing* (bị san phẳng thuộc tính) và *Over-squashing* (bị nén thông tin quá mức) bằng việc liên kết động trực tiếp tất cả các nút bất kể khoảng cách cấu trúc giải phẫu.
* Kiểm chứng thực nghiệm giải thuật **FastGTN** (sinh meta-path động) trên tập dữ liệu đồ thị dị thể học thuật **ACM** và đồ thị phân tích hành vi mạng lưới **Yelp_Business**.
* Đánh giá hiện tượng **"Hardware Lottery"** (Xổ số phần cứng) — lý giải vì sao Transformer thống trị thế giới thực dụng nhờ khả năng tính toán ma trận dày (Dense Matrix) song song hóa cực tốt trên GPU.

---

## 📁 2. Cấu trúc Kho lưu trữ
Kho lưu trữ chứa toàn bộ mã nguồn xử lý và các file vá lỗi hệ thống (`main.py`) giúp tương thích với phiên bản PyTorch Geometric mới nhất:

```text
├── main.py            # Script huấn luyện chính (Đã được vá lỗi f1_score từ PyG)
├── model_gtn.py       # Định nghĩa kiến trúc Graph Transformer Networks (GTN) chuẩn
├── model_fastgtn.py  # Định nghĩa kiến trúc FastGTN tối ưu hóa tốc độ và bộ nhớ
├── layer.py           # Triển khai lớp sinh meta-path động cho mạng GTN
├── layer_fastgtn.py   # Triển khai lớp FastGTNLayer và Non-Local SkipConnection
├── utils.py           # Các hàm bổ trợ chuẩn hóa ma trận kề đối xứng (_norm)
├── inits.py           # Khởi tạo trọng số Xavier/Glorot và Zeros cho Tensor



📊 3. Hướng dẫn Chuẩn bị Dữ liệu đầu vàoDữ liệu đồ thị dị thể học thuật và thương mại cần được đồng bộ hóa cấu trúc liên kết và nhãn về không gian nhúng chung trước khi đưa vào mô hình.Tập dữ liệu ACM (Mặc định)Đảm bảo bạn đặt file ma trận gốc ACM.mat vào đúng cấu trúc thư mục xử lý. Sau khi tiền xử lý, hệ thống sẽ trích xuất:node_features.pkl: Ma trận thuộc tính cấu trúc nút kích thước (9172, 1903).edges.pkl: Danh sách 4 ma trận kề thưa đại diện cho các quan hệ hai chiều: Paper $\rightarrow$ Author, Author $\rightarrow$ Paper, Paper $\rightarrow$ Subject, Subject $\rightarrow$ Paper.labels.pkl: Mảng nhãn phân tách tập Train (19.8%), Val (9.9%), Test (70.3%) phục vụ tác vụ phân loại chủ đề bài báo.Tập dữ liệu Yelp_Business (Mở rộng)Đồ thị phân tích hành vi thương mại chứa mạng lưới quy mô lớn hơn với 31.035 nút và 875.786 cạnh.
Dữ liệu được trích xuất từ cấu trúc file thô chứa các quan hệ: User-Business, Business-Category, và User-User.
└── data/              # Thư mục chứa dữ liệu đầu vào sau khi tiền xử lý (.pkl)
    ├── ACM/           # Dữ liệu đồ thị dị thể học thuật ACM (9.172 nút)
    └── Yelp_Business/ # Dữ liệu đồ thị dị thể Yelp (31.035 nút)

📈 4. Kết quả Phân tích & Đánh giá Hiệu năng
1. Kết quả trên đồ thị ACM
Mô hình FastGTN thể hiện khả năng khớp dữ liệu cực kỳ mạnh mẽ trên tập huấn luyện nhờ cơ chế Attention tự học meta-path:

Train Macro-F1 / Micro-F1: Đạt mức tối ưu hóa cao vượt trội >94.7%.

Test Macro-F1 / Micro-F1: Đạt hiệu năng ổn định lần lượt là 58.11% và 61.00%. Thử nghiệm này gặp thách thức lớn do đặc trưng đồ thị ACM thưa thớt cực đoan và tỷ lệ dữ liệu gán nhãn tập Train rất thấp (~19.8%).

2. So sánh hiệu năng với Baseline (Không sử dụng cấu trúc đồ thị)
Khi đối chiếu với kiến trúc mạng MLP thuần túy (chỉ nhìn thấy thuộc tính tĩnh, không có cơ chế truyền tin qua đồ thị), Graph Transformer mang lại bước nhảy vọt về chỉ số F1-Score:

Yelp_Business: Chỉ số Macro-F1 cải thiện +0.0221 (Tăng từ 0.4513 lên 0.4734).

MovieLens 100K: Chỉ số Macro-F1 cải thiện từ 0.5222 lên 0.5545 (Gain: +0.0323).
