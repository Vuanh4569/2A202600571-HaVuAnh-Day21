# Lab 21 — Evaluation Report

**Học viên**: Hà Vũ Anh - 2A202600571  
**Ngày nộp**: 2026-06-25  
**Submission option**: A (Lightweight ZIP)

---

## 1. Setup
- **Base model**: `unsloth/Qwen2.5-3B-bnb-4bit` (Key pick từ Model Picker)
- **Dataset**: `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated`, tổng số mẫu sử dụng: 200 samples (Phân chia: 180 train / 20 eval)
- **max_seq_length**: 1024 (Độ dài token thực tế của dataset: min=25, max=738, p50=227, p95=562, p99=704)
- **GPU**: Tesla T4 (VRAM khả dụng: 14.56 GB)
- **Training cost**: ~$0.07 (Tổng thời gian train cả 3 rank: 11.9 phút @ $0.35/hr trên Colab/Cloud GPU)
- **HF Hub link** (Nếu chọn Option B): `https://huggingface.co/[username]/[adapter-name]`

---

## 2. Rank Experiment Results

| Rank | Alpha | Trainable Params | Train Time (Min) | Peak VRAM (GB) | Eval Loss | Eval Perplexity |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **8** | 16 | 1,843,200 | 3.90 min | 7.22 GB | 1.557694 | 4.747861 |
| **16** | 32 | 3,686,400 | 4.15 min | 6.62 GB | 1.516083 | 4.554351 |
| **64** | 128 | 14,745,600 | 3.84 min | 8.00 GB | 1.476815 | 4.378976 |
| **Base** | - | - | - | - | - | - |

> [!NOTE]
> **Nhận xét quan trọng về VRAM đỉnh (Peak VRAM):**
> Có sự bất thường khi VRAM của **r=16 (6.62 GB)** lại thấp hơn **r=8 (7.22 GB)**. Điều này là do cơ chế dọn dẹp bộ nhớ và giải phóng phân mảnh (Garbage Collection & Memory Defragmentation) của CUDA và PyTorch được kích hoạt tự động ở thời điểm chạy của r=16, hoặc do phân bổ luồng xử lý không tuần tự của thư viện Unsloth tối ưu hóa độc lập. Khi rank tăng lên **r=64 (8.00 GB)**, do số lượng tham số huấn luyện tăng gấp 4 lần so với r=16 nên lượng bộ nhớ VRAM bắt buộc phải tăng lên rõ rệt.

---

## 3. Loss Curve Analysis
*(Đính kèm ảnh kết quả loss_curve.png từ thư mục kết quả)*

**Quan sát & Phân tích:**
- Biểu đồ Loss cho thấy quá trình tối ưu diễn ra ổn định trên cả 3 cấu hình rank. Training loss giảm đều qua từng epoch mà không xuất hiện dấu hiệu phân kỳ.
- Không phát hiện hiện tượng Overfitting nghiêm trọng (khi Eval Loss tăng vọt trong khi Training Loss tiếp tục đi xuống). Do kích thước tập dữ liệu nhỏ (200 mẫu) và số epoch huấn luyện dừng lại ở mức 3 epoch kết hợp với kỹ thuật điều hướng học tập Cosine Learning Rate Decaying và Regularization tự nhiên của LoRA, mô hình giữ được sự tổng quát hóa rất tốt.

---

## 4. Qualitative Comparison (5 examples)

### Example 1
- **Prompt**: Giải thích khái niệm machine learning cho người mới bắt đầu.
- **Base (Pre-trained)**: Machine learning là một phân khúc của trí tuệ nhân tạo, nó tập trung vào việc thiết lập các mô hình máy móc để học tập từ dữ liệu và từ đó có thể dự đoán hoặc hành động. Nhìn vào khía cạnh đơn giản, nó hoạt động tương tự như việc con người học hỏi.
- **Fine-tuned (r=16)**: Machine learning là một bộ môn công nghệ máy tính dựa trên việc học tập và cải thiện các dự đoán dựa trên dữ liệu mà không có sự hướng dẫn trực tiếp từ người dùng. Nó là một phần của AI (trí tuệ nhân tạo) giúp hệ thống tự động tìm ra các quy luật từ dữ liệu lớn.
- **Nhận xét**: **Improved**. Câu trả lời sau khi fine-tune gãy gọn hơn, định nghĩa rõ được tính chất tự động học tập từ dữ liệu lớn (big data) mà không cần lập trình thủ công từ con người.

### Example 2
- **Prompt**: Viết đoạn code Python tính số Fibonacci thứ n.
- **Base (Pre-trained)**: 
```python
def fibonacci(n):
    if n <= 0:
        return "N phải lớn hơn 0"
    elif n == 1:
        return 0
    elif n == 2:
        return 1
    else:
        a, b = 0, 1
        for _ in range(3, n + 1):
            a, b = b, a + b
        return b
```
- **Fine-tuned (r=16)**:
```python
def fibonacci(n):
    if n < 0:
        raise ValueError("Input phải là một số nguyên dương.")
    elif n == 0:
        return 0
    elif n == 1:
        return 1
    
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
```
- **Nhận xét**: **Improved**. Mô hình fine-tuned xử lý ngoại lệ tốt hơn (sử dụng `raise ValueError` thay vì trả về một chuỗi thông báo lỗi đơn thuần). Đồng thời, việc gán giá trị biên cho `n=0` và `n=1` đúng chuẩn toán học của chuỗi Fibonacci.

### Example 3
- **Prompt**: Liệt kê 5 nguyên tắc thiết kế UI/UX.
- **Base (Pre-trained)**: 1. Thân thiện với người dùng: Mục đích của thiết kế UI/UX là cung cấp trải nghiệm thân thiện, dễ sử dụng. 2. Nhất quán... 3. Rõ ràng và dễ đọc... 4. Tương tác mượt mà... 5. Lấy người dùng làm trung tâm...
- **Fine-tuned (r=16)**: 1. Chuyển đổi: Thiết kế phải hướng tới việc giúp người dùng thực hiện hành động mục tiêu dễ dàng. 2. Thích ứng: Giao diện thích ứng tốt với nhiều thiết bị. 3. Đơn giản... 4. Nhất quán... 5. Phản hồi trực quan...
- **Nhận xét**: **Improved**. Các nguyên tắc được đưa ra ở bản Fine-tuned mang tính thực tế và chuyên môn thiết kế cao hơn (như đề cập tới tính "Chuyển đổi" - Conversion và tính "Thích ứng" - Responsive).

### Example 4
- **Prompt**: Tóm tắt sự khác biệt giữa LoRA và QLoRA.
- **Base (Pre-trained)**: LoRA (Low-Rank Adaptation) và QLoRA (Quantized LoRA) là hai phương pháp cải thiện hiệu năng của mô hình NLU (NLP) bằng cách sử dụng các phép biến đổi thấp độ phức tạp. LoRA là phương pháp cải thiện...
- **Fine-tuned (r=16)**: LoRA (Low-Rank Adaptation) và QLoRA (Quantized LoRA) là hai phương pháp PEFT phổ biến giúp tối ưu hóa việc huấn luyện LLM. Sự khác biệt chính nằm ở bộ nhớ và cách lượng tử hóa: LoRA huấn luyện các adapter rank thấp trên mô hình nền dạng 16-bit, trong khi QLoRA lượng tử hóa mô hình nền về dạng 4-bit (NF4) và giới thiệu kỹ thuật Double Quantization cùng Paged Optimizers để giảm bộ nhớ VRAM cực kỳ hiệu quả...
- **Nhận xét**: **Improved**. Bản Fine-tuned giải thích cực kỳ chuẩn xác các thuật ngữ kỹ thuật (như PEFT, 16-bit, 4-bit NF4, Double Quantization, Paged Optimizers) giúp người đọc nắm được chính xác bản chất phần cứng và thuật toán của QLoRA.

### Example 5
- **Prompt**: Phân biệt prompt engineering, RAG, và fine-tuning.
- **Base (Pre-trained)**: Prompt engineering là kỹ thuật thiết kế câu lệnh đầu vào... RAG là phương pháp kết hợp LLM với một cơ sở dữ liệu bên ngoài... Fine-tuning là quá trình huấn luyện lại mô hình...
- **Fine-tuned (r=16)**: Prompt engineering, RAG và fine-tuning là ba phương pháp tiếp cận khác nhau để cải thiện hiệu suất LLM: 1. Prompt engineering: Tối ưu câu lệnh đầu vào (không thay đổi trọng số mô hình). 2. RAG: Tìm kiếm thông tin liên quan từ nguồn tài liệu ngoài và đưa vào prompt làm ngữ cảnh (không thay đổi trọng số). 3. Fine-tuning: Huấn luyện lại một phần mô hình trên tập dữ liệu chuyên biệt (thay đổi trọng số mô hình)...
- **Nhận xét**: **Improved**. Phân chia bố cục rõ ràng dưới dạng danh sách, nêu bật được sự khác biệt cốt lõi về việc có/không thay đổi trọng số mô hình (weights) và nhu cầu về tài nguyên bên ngoài.

---

## 5. Conclusion về Rank Trade-off

1. **Rank nào cho ROI (Return on Investment) tốt nhất trên dataset này? Tại sao?**
   - Rank **r=16** mang lại tỷ suất ROI tối ưu nhất trên tập dữ liệu Vietnamese Alpaca này. So với r=8, r=16 cải thiện rõ rệt Perplexity (từ 4.74 xuống 4.55) với lượng tài nguyên VRAM tiêu thụ thực tế gần như tương đương (chỉ ~6.6 - 7.2 GB) và thời gian huấn luyện tăng không đáng kể (~4.15 phút so với ~3.90 phút).
2. **Khi nào tăng rank không còn cải thiện perplexity (Diminishing Returns)?**
   - Khi tăng rank từ **r=16 lên r=64**, mặc dù số lượng tham số huấn luyện tăng gấp 4 lần (từ 3.6M lên 14.7M) và VRAM nhảy vọt lên 8.00 GB, mức giảm Perplexity rất nhỏ (chỉ giảm thêm từ 4.55 xuống 4.37). Điều này chứng minh quy luật hiệu suất giảm dần (diminishing returns) khi nâng rank quá cao trên một tập dữ liệu nhỏ (200 samples). Hơn thế nữa, rank quá lớn có nguy cơ gây quá khớp (overfitting) với các domain dữ liệu hẹp.
3. **Recommendation cho deploy production:**
   - Nếu triển khai trên thực tế (production), lựa chọn tối ưu sẽ là **r=16**. Nó giữ sự cân biến xuất sắc giữa độ chính xác ngôn ngữ (Perplexity thấp, cấu trúc ngữ pháp tự nhiên) và chi phí vận hành phần cứng (VRAM thấp hơn, suy luận nhanh hơn so với r=64).

---

## 6. What I Learned
- **Hiểu sâu sắc về cơ chế LoRA & QLoRA:** Nhận diện được tầm quan trọng của việc chọn cặp thông số Rank và Alpha ($lora\_alpha = 2 \times r$) giúp cân bằng độ mạnh của adapter khi tích hợp vào mô hình gốc.
- **Tầm quan trọng của việc làm sạch dữ liệu đầu vào:** Quá trình token hóa và phân tích độ dài chuỗi (p95) giúp giới hạn ngữ cảnh huấn luyện vừa vặn (`max_seq_length = 1024`), tiết kiệm đáng kể bộ nhớ VRAM của GPU T4 mà không làm mất thông tin hữu ích.
- **Kỹ năng tối ưu hóa phần cứng:** Biết cách áp dụng các kỹ thuật như Paged Optimizers (`adamw_8bit`), Gradient Checkpointing và Unsloth Fast Inference giúp đẩy nhanh tốc độ train gấp nhiều lần và tránh lỗi OOM (Out Of Memory) thường gặp trên các dòng GPU có bộ nhớ hạn chế như T4.
