import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple

# 尝试导入视觉和下载库
try:
    import cv2
    import torch
    import yt_dlp
    from transformers import CLIPProcessor, CLIPModel
    from PIL import Image
    from googleapiclient.discovery import build
except ImportError as e:
    print(f"缺少必要依赖: {e}")
    print("请执行: pip install google-api-python-client yt-dlp opencv-python torch transformers pillow")
    sys.exit(1)

# --- 配置区 ---
DEFAULT_QUERIES = [
    "combat trauma wound",
    "battlefield medic treating wound",
    "war documentary field hospital injured",
    "战场救护 伤员 真实记录",
]

# CLIP 模型的正向和负向提示词
POSITIVE_PROMPTS = [
    "a close-up of an open wound with blood",
    "a medical training video frame showing a visible traumatic wound",
    "a combat casualty care scene with bleeding injury",
    "a visible laceration, burn, gunshot wound, or blast wound",
]

NEGATIVE_PROMPTS = [
    "a person talking with no visible injury",
    "a military vehicle, weapon, or battlefield scene with no visible wound",
    "a doctor or medic standing in a classroom",
    "a diagram, map, or text slide",
]

class VLMDetector:
    """视觉大模型检测器 (使用 CLIP)"""
    def __init__(self, device="auto", batch_size=8):
        self.batch_size = batch_size
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps" # Mac M芯片加速
            else:
                self.device = "cpu"
        else:
            self.device = device
            
        print(f"[INFO] 正在加载 CLIP 模型到设备: {self.device} ... (首次运行可能需要下载模型)")
        model_name = "openai/clip-vit-base-patch32"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()
        self.prompts = POSITIVE_PROMPTS + NEGATIVE_PROMPTS
        self.positive_count = len(POSITIVE_PROMPTS)

    def score_frames(self, frames_bgr: List[Any]) -> List[float]:
        images = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames_bgr]
        inputs = self.processor(text=self.prompts, images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)
            # 将所有正向提示词的概率相加作为该帧的"创伤得分"
            positive_scores = probs[:, :self.positive_count].sum(dim=1)
            
        return [float(x) for x in positive_scores.cpu().tolist()]

def youtube_search(api_key: str, queries: List[str], max_per_query: int) -> List[Dict[str, Any]]:
    """调用 YouTube API 搜索视频"""
    youtube = build("youtube", "v3", developerKey=api_key)
    seen_vids = {}

    for query in queries:
        try:
            resp = youtube.search().list(
                part="snippet", q=query, type="video", maxResults=max_per_query, safeSearch="none"
            ).execute()
            
            for item in resp.get("items", []):
                vid = item["id"]["videoId"]
                if vid not in seen_vids:
                    seen_vids[vid] = {
                        "video_id": vid,
                        "title": item["snippet"]["title"],
                        "url": f"https://www.youtube.com/watch?v={vid}"
                    }
        except Exception as e:
            print(f"[WARN] 搜索 '{query}' 时发生错误: {e}")

    return list(seen_vids.values())

def download_temp_video(url: str, temp_dir: Path) -> str:
    """使用 yt-dlp 下载超低分辨率视频，加快下载和推理速度"""
    ydl_opts = {
        'format': 'worstvideo[ext=mp4]/worst', # 获取最差画质，节省带宽
        'outtmpl': str(temp_dir / '%(id)s.%(ext)s'),
        'quiet': True,
        'no_warnings': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return ydl.prepare_filename(info)

def find_visual_segments(video_path: str, detector: VLMDetector, threshold=0.4, sample_fps=1.0) -> List[Dict]:
    """逐帧扫描视频，找出连续高分的片段"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened() or fps <= 0:
        return []

    frame_step = int(fps / sample_fps)
    frame_idx = 0
    time_scores = []
    
    batch_frames = []
    batch_times = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx % frame_step == 0:
            current_time = frame_idx / fps
            batch_frames.append(frame)
            batch_times.append(current_time)
            
            if len(batch_frames) >= detector.batch_size:
                scores = detector.score_frames(batch_frames)
                for t, s in zip(batch_times, scores):
                    time_scores.append({"time": t, "score": s})
                batch_frames = []
                batch_times = []
                
        frame_idx += 1

    # 处理剩余的帧
    if batch_frames:
        scores = detector.score_frames(batch_frames)
        for t, s in zip(batch_times, scores):
            time_scores.append({"time": t, "score": s})

    cap.release()

    # 将离散的高分帧融合成片段 (最大 30 秒)
    segments = []
    current_seg = None
    
    for item in time_scores:
        if item["score"] >= threshold:
            if current_seg is None:
                # 往前缓冲 2 秒
                start_time = max(0, item["time"] - 2)
                current_seg = {"start": start_time, "end": item["time"], "max_score": item["score"]}
            else:
                current_seg["end"] = item["time"]
                current_seg["max_score"] = max(current_seg["max_score"], item["score"])
                
                # 截断：如果片段长度超过 30 秒，强行切断并保存
                if current_seg["end"] - current_seg["start"] >= 30:
                    # 往后缓冲 2 秒
                    current_seg["end"] += 2
                    segments.append(current_seg)
                    current_seg = None
        else:
            if current_seg is not None:
                # 允许 3 秒内的高分断层（防止画面中偶尔闪过其他东西导致片段断裂）
                if item["time"] - current_seg["end"] > 3:
                    current_seg["end"] += 2
                    # 过滤掉低于 3 秒的无效闪烁片段
                    if current_seg["end"] - current_seg["start"] >= 3:
                        segments.append(current_seg)
                    current_seg = None

    if current_seg is not None and (current_seg["end"] - current_seg["start"] >= 3):
        current_seg["end"] += 2
        segments.append(current_seg)

    return segments

def main():
    parser = argparse.ArgumentParser(description="VLM Vision-based YouTube War Trauma Clip Finder")
    parser.add_argument("--api-key", default=os.environ.get("YOUTUBE_API_KEY", ""), help="YouTube API Key")
    parser.add_argument("--max-per-query", type=int, default=5, help="每个搜索词处理多少个视频")
    parser.add_argument("--threshold", type=float, default=0.45, help="CLIP 模型的识别阈值 (0-1)")
    parser.add_argument("--out-csv", default="vlm_trauma_clips.csv", help="输出 CSV 文件名")
    args = parser.parse_args()

    if not args.api_key:
        print("错误：缺少 YouTube API Key。请设置环境变量或传入 --api-key。")
        sys.exit(1)

    temp_dir = Path("./temp_vids")
    temp_dir.mkdir(exist_ok=True)

    print("[INFO] 正在初始化 VLM 模型...")
    detector = VLMDetector()

    print("[INFO] 开始搜索 YouTube 视频...")
    videos = youtube_search(args.api_key, DEFAULT_QUERIES, args.max_per_query)
    print(f"[INFO] 找到 {len(videos)} 个视频用于视觉扫描。")

    results = []

    for idx, vid_info in enumerate(videos, 1):
        vid_url = vid_info["url"]
        title = vid_info["title"]
        print(f"\n[{idx}/{len(videos)}] 正在处理: {title[:50]}...")
        
        temp_file = ""
        try:
            print("  -> 下载低分辨率视频流用于分析...")
            temp_file = download_temp_video(vid_url, temp_dir)
            
            print("  -> VLM 逐帧视觉扫描中...")
            segments = find_visual_segments(temp_file, detector, threshold=args.threshold)
            
            if not segments:
                print("  [SKIP] 未发现明显战创伤画面。")
            else:
                for i, seg in enumerate(segments):
                    start_s = int(seg['start'])
                    end_s = int(seg['end'])
                    timestamp_url = f"{vid_url}&t={start_s}s"
                    print(f"  [HIT] 发现片段 {i+1}: {start_s}s - {end_s}s (得分: {seg['max_score']:.2f}) -> {timestamp_url}")
                    
                    results.append({
                        "video_id": vid_info["video_id"],
                        "title": title,
                        "url": vid_url,
                        "timestamp_url": timestamp_url,
                        "start_seconds": start_s,
                        "end_seconds": end_s,
                        "duration": end_s - start_s,
                        "max_score": round(seg['max_score'], 3)
                    })
        except Exception as e:
            print(f"  [ERROR] 处理失败: {e}")
        finally:
            # 清理临时文件，防止占用硬盘
            if temp_file and os.path.exists(temp_file):
                os.remove(temp_file)

    # 写入结果
    if results:
        keys = results[0].keys()
        with open(args.out_csv, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)
        print(f"\n[DONE] 分析完成！共找到 {len(results)} 个符合条件的视频片段，已保存至 {args.out_csv}")
    else:
        print("\n[DONE] 分析完成，未找到任何符合条件的片段。可以尝试降低 --threshold 参数。")

if __name__ == "__main__":
    main()