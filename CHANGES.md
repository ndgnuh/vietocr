# Thay đổi so với bản gốc

Thêm một số backbone (chỉ Inception có pretrain)
	- Inception
	- MobileNet (V3L, V3M, V2)
	- ResNet (18, 34, 50, ...)
	- EfficientNetV1 (B1-B7)
	- ShuffleNet V2

Thay đổi cách chạy mô hình
	- Dùng forward pass để chạy sequence to sequence cho cả training và inference (thay vì teacher forcing như bản gốc)
	- Bỏ beamsearch

Một vài thay đổi khác tới config và seq2seq head
