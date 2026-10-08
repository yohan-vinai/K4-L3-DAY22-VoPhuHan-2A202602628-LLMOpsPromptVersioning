# RAGAS: so sánh prompt V1 và V2

| Chỉ số | V1 | V2 | Nhận xét |
|---|---:|---:|---|
| Faithfulness | 0.9837 | 0.9888 | V2 cao hơn 0.0051; cả hai đều vượt 0.9. |
| Answer relevancy | 0.9048 | 0.8968 | V1 cao hơn 0.0080. |
| Context recall | 1.0000 | 1.0000 | Hai phiên bản bằng nhau. |
| Context precision | 0.9617 | 0.9617 | Hai phiên bản bằng nhau. |

V1 hướng tới câu trả lời ngắn gọn nên có thể giữ câu trả lời bám sát câu hỏi hơn, phù hợp với answer relevancy nhỉnh hơn. V2 yêu cầu đọc kỹ facts và trình bày có cấu trúc; faithfulness nhỉnh hơn một chút. Đây là cách diễn giải phù hợp với prompt và các điểm đo, chưa đủ để kết luận prompt là nguyên nhân duy nhất. Cả hai đạt context recall 1.0 và có context precision như nhau, nên bộ context truy xuất bao phủ reference tốt trong các mẫu đã chấm.
