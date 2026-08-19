"""
图像处理模块 — 多模态 RAG 的核心

职责：
1. 用 CLIP 对图像生成 Embedding（用于图文混合检索）
2. 用 VLM 对图像生成文本描述（用于LLM理解图像内容）
3. 将图像描述 + CLIP向量 注入向量数据库

设计理由：
- CLIP Embedding 让用户可以用文本检索到相关图像（跨模态）
- VLM 描述让 LLM 能"理解"图像内容（图像→文本→生成）
- 两者结合：检索用CLIP，生成用VLM描述

技术栈：
- CLIP: openai/clip-vit-base-patch32（图文对齐编码）
- VLM: Qwen-VL-Chat 或 fallback 到简单描述
"""
import os
from pathlib import Path
from typing import List, Dict, Optional
import hashlib


class ImageProcessor:
    def __init__(
        self,
        clip_model: str = "openai/clip-vit-base-patch32",
        vlm_wrapper=None,
        device: str = "cpu",
    ):
        self.clip_model_name = clip_model
        self.vlm = vlm_wrapper
        self.device = device
        self.clip_processor = None
        self.clip_model = None
        self._load_clip()

    def _load_clip(self):
        """加载 CLIP 模型"""
        print(f"加载 CLIP 模型: {self.clip_model_name}")
        try:
            from transformers import CLIPProcessor, CLIPModel
            import torch
            self.clip_processor = CLIPProcessor.from_pretrained(self.clip_model_name)
            self.clip_model = CLIPModel.from_pretrained(self.clip_model_name)
            self.clip_model.to(self.device)
            self.clip_model.eval()
            print("CLIP 加载完成")
        except Exception as e:
            print(f"⚠️ CLIP 加载失败（多模态检索将不可用）: {e}")
            self.clip_processor = None
            self.clip_model = None

    def encode_image(self, image_path: str) -> Optional[List[float]]:
        """用 CLIP 编码图像为向量"""
        if self.clip_model is None or self.clip_processor is None:
            return None

        try:
            from PIL import Image
            import torch
            img = Image.open(image_path).convert("RGB")
            inputs = self.clip_processor(images=img, return_tensors="pt").to(self.device)
            with torch.no_grad():
                features = self.clip_model.get_image_features(**inputs)
            embedding = features.cpu().numpy()[0].tolist()
            return embedding
        except Exception as e:
            print(f"  图像编码失败 {image_path}: {e}")
            return None

    def encode_text(self, text: str) -> Optional[List[float]]:
        """用 CLIP 编码文本为向量（用于跨模态检索）"""
        if self.clip_model is None or self.clip_processor is None:
            return None

        try:
            import torch
            inputs = self.clip_processor(text=[text], return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                features = self.clip_model.get_text_features(**inputs)
            embedding = features.cpu().numpy()[0].tolist()
            return embedding
        except Exception as e:
            print(f"  文本CLIP编码失败: {e}")
            return None

    def describe_image(self, image_path: str) -> str:
        """用 VLM 生成图像描述"""
        if self.vlm is not None:
            try:
                prompt = (
                    "请描述这张学术文档中的图像内容。如果是图表，请说明：\n"
                    "1. 图表类型（折线图/柱状图/散点图/架构图/流程图/公式/其他）\n"
                    "2. 主要内容（坐标轴、数据趋势、关键组件等）\n"
                    "3. 图表说明的关键信息或结论\n"
                    "请用100字以内简洁描述。"
                )
                description = self.vlm.generate(prompt + "\n[图像路径: " + image_path + "]")
                return description.strip()
            except Exception as e:
                print(f"  VLM描述失败: {e}")

        meta_tag = Path(image_path).stem
        return f"[图像内容：{meta_tag}，未能生成详细描述]"

    def process_image_chunk(self, chunk: Dict) -> Dict:
        """处理图像chunk：生成描述 + CLIP编码"""
        image_path = chunk.get("image_path")
        if not image_path or not Path(image_path).exists():
            return chunk

        description = self.describe_image(image_path)
        clip_embedding = self.encode_image(image_path)

        enriched = chunk.copy()
        enriched["content"] = f"[图像描述] {description}"
        enriched["image_description"] = description
        enriched["clip_embedding"] = clip_embedding
        enriched["metadata"]["modality"] = "image"
        enriched["metadata"]["has_description"] = True

        return enriched

    def process_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """批量处理所有chunk（文本chunk不变，图像chunk加描述+编码）"""
        text_count = sum(1 for c in chunks if c["metadata"].get("modality", "text") == "text")
        image_count = sum(1 for c in chunks if c["metadata"].get("modality") == "image")
        print(f"处理 chunks: {text_count} 文本 + {image_count} 图像")

        processed = []
        for i, chunk in enumerate(chunks):
            if chunk["metadata"].get("modality") == "image":
                print(f"  [{i+1}] 生成图像描述: {chunk['metadata'].get('image_name', '?')}")
                enriched = self.process_image_chunk(chunk)
                processed.append(enriched)
            else:
                processed.append(chunk)

        return processed

    def is_available(self) -> bool:
        """CLIP 是否可用"""
        return self.clip_model is not None
