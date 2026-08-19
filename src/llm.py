"""
LLM 模块 — 本地大模型加载与生成

支持：
1. Qwen2.5-7B-Chat（推荐，中文生成质量好）
2. 其他 HuggingFace 兼容模型

关键设计：使用 chat_template 标准化输入格式，解决小模型指令泄露问题。
"""
import torch
from typing import Optional


class LLMWrapper:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        load_in_4bit: bool = False,
        max_new_tokens: int = 512,
    ):
        self.model_path = model_path
        self.device = device
        self.load_in_4bit = load_in_4bit
        self.max_new_tokens = max_new_tokens
        self.model = None
        self.tokenizer = None
        self._load()

    def _load(self):
        from transformers import AutoTokenizer, AutoModelForCausalLM

        print(f"加载 LLM: {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )

        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
            )

        self.model.eval()
        print(f"LLM 加载完成 (4bit={self.load_in_4bit})")

    def generate(self, prompt: str, history: list = None, max_new_tokens: int = None) -> str:
        """生成回答（使用 chat_template 标准化格式）"""
        max_tokens = max_new_tokens or self.max_new_tokens

        messages = [{"role": "system", "content": "你是一个学术文档问答助手。"}]
        if history:
            for h in history[-3:]:
                if "question" in h:
                    messages.append({"role": "user", "content": h["question"]})
                    messages.append({"role": "assistant", "content": h["answer"]})
        messages.append({"role": "user", "content": prompt})

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
            )

        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        return response.strip()

    def generate_simple(self, prompt: str, max_new_tokens: int = 100) -> str:
        """简单生成（用于查询改写等短文本任务）"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        return self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()
