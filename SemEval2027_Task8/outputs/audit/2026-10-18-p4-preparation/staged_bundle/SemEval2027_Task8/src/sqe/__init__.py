"""Code dùng chung cho các notebook 03–08 (SemEval-2027 Task 8, EN, Subtask A + B).

    data.py        đọc TSV + gắn fold, dựng input (comment / cặp câu / ghép thẳng)
    metrics.py     macro-F1 từng fold, so sánh cặp hai cấu hình
    results.py     ghi bảng kết quả chung + lưu/đọc OOF
    classical.py   TF-IDF + Logistic Regression (giống notebook 02)
    train.py       fine-tune transformer theo 5 fold (một hoặc hai đầu ra)
    llm.py         gọi LLM qua Ollama (zero-shot / few-shot)
    submission.py  ghi file nộp Codabench và kiểm tra format
"""
