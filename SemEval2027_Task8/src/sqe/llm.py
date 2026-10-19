"""Gọi LLM chạy cục bộ qua Ollama (http://localhost:11434) để phân loại zero-shot / few-shot.

* Định nghĩa nhãn lấy gần nguyên văn từ guideline (mục 3.1, 3.2).
* temperature = 0 (giống baseline Qwen2.5-7B của ban tổ chức) → kết quả lặp lại được.
* Dùng "structured output" của Ollama: ép model trả JSON {"label": <một nhãn hợp lệ>},
  nên không phải đoán nhãn từ một đoạn văn tự do.
* Mỗi câu trả lời được ghi ngay vào file .jsonl → bị ngắt giữa chừng thì chạy lại sẽ tiếp tục.
"""
import json
import time

import requests

from .data import OUT_DIR, TASKS

OLLAMA = "http://localhost:11434"
CACHE_DIR = OUT_DIR / "llm_cache"

DEFINITIONS = {
    "A": (
        "Task: decide whether the YouTube comment contains a STEREOTYPE targeting LGBTQIA+ people.\n"
        "A stereotype is a generalized, oversimplified, or fixed belief about a group of people.\n"
        "Labels:\n"
        "- yes: the comment contains a stereotype targeting LGBTQIA+ people\n"
        "- no: the comment does not contain such a stereotype"
    ),
    "B": (
        "Task: classify whether the YouTube comment is HATE SPEECH against LGBTQIA+ people.\n"
        "Labels:\n"
        "- yes_explicit: explicit hateful content, including direct slurs, insults, or other overtly "
        "hateful expressions targeting LGBTQIA+ individuals or groups\n"
        "- yes_implicit: hateful content expressed indirectly, implicitly, or through coded language, "
        "including implied insults, derogatory stereotypes, and other forms of indirect hateful expression\n"
        "- no: neutral, supportive, or otherwise non-hateful content"
    ),
}

SYSTEM = ("You are an expert annotator of online content about the LGBTQIA+ community. "
          "Interpret the comment together with the video title and description. "
          "Answer only with the requested JSON.")


def format_item(row, with_desc=True, max_desc_chars=600):
    desc = row["yt_description"][:max_desc_chars]
    parts = [f"Video title: {row['yt_title']}"]
    if with_desc:
        parts.append(f"Video description: {desc}")
    parts.append(f"Comment: {row['yt_comment']}")
    return "\n".join(parts)


def build_messages(row, subtask, shots=None, with_desc=True):
    """shots: danh sách (row, nhãn) làm ví dụ few-shot (lấy từ các fold KHÁC fold đang chấm)."""
    msgs = [{"role": "system", "content": SYSTEM}]
    task = DEFINITIONS[subtask] + '\nReturn JSON: {"label": "<one of the labels>"}'
    for ex, lab in shots or []:
        # ví dụ few-shot: cắt description ngắn hơn (200 ký tự) để prompt không quá dài
        msgs.append({"role": "user", "content": task + "\n\n" + format_item(ex, with_desc, 200)})
        msgs.append({"role": "assistant", "content": json.dumps({"label": lab})})
    msgs.append({"role": "user", "content": task + "\n\n" + format_item(row, with_desc)})
    return msgs


def server_models():
    r = requests.get(f"{OLLAMA}/api/tags", timeout=5)
    r.raise_for_status()
    return [m["name"] for m in r.json()["models"]]


def classify(row, subtask, model, shots=None, with_desc=True, timeout=300):
    labels = TASKS[subtask][1]
    schema = {"type": "object", "properties": {"label": {"type": "string", "enum": labels}},
              "required": ["label"]}
    body = {"model": model, "messages": build_messages(row, subtask, shots, with_desc),
            "format": schema, "stream": False,
            "options": {"temperature": 0, "seed": 42, "num_ctx": 4096}}
    r = requests.post(f"{OLLAMA}/api/chat", json=body, timeout=timeout)
    r.raise_for_status()
    text = r.json()["message"]["content"]
    try:
        lab = json.loads(text)["label"]
    except (json.JSONDecodeError, KeyError, TypeError):
        lab = None
    return lab if lab in labels else None, text


def run(df, subtask, model, run_name, shots_for=None, with_desc=True, log=print, every=50):
    """Phân loại mọi dòng của df; trả về {id: nhãn}. shots_for(row) → danh sách ví dụ few-shot."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{run_name}_{subtask}.jsonl"
    done = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            done[rec["id"]] = rec["label"]
    todo = df[~df["StereoQueerEval_id"].isin(done)]
    log(f"{run_name}/{subtask}: đã có {len(done)}, còn {len(todo)}")
    t0 = time.time()
    with open(path, "a", encoding="utf-8") as f:
        for i, (_, row) in enumerate(todo.iterrows(), 1):
            shots = shots_for(row) if shots_for else None
            lab, raw = classify(row, subtask, model, shots, with_desc)
            rec = {"id": row["StereoQueerEval_id"], "label": lab, "raw": raw}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            done[rec["id"]] = lab
            if i % every == 0:
                rate = (time.time() - t0) / i
                log(f"  {i}/{len(todo)}  ({rate:.1f} s/mẫu, còn ~{rate * (len(todo) - i) / 60:.0f} phút)")
    return done
