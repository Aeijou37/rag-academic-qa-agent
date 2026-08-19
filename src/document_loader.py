"""
文档加载模块 — 支持 PDF / DOCX / TXT / MD + 图像提取

职责：
1. 解析不同格式的文档，提取干净文本，附带元数据
2. 从 PDF 中提取图片（多模态 RAG 的图像来源）
3. 为每张图片生成图像描述（由 image_processor 模块完成）
"""
from pathlib import Path
from typing import List, Dict, Optional
import hashlib
import io


SUPPORTED_FORMATS = {".pdf", ".docx", ".txt", ".md"}
IMAGE_FORMATS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
MIN_IMAGE_SIZE = 100  # 小于此尺寸的图片跳过（可能是图标/装饰）


class DocumentLoader:
    def __init__(self):
        self.documents: List[Dict] = []

    def load(self, file_path: str) -> List[Dict]:
        """加载单个文件，返回文档块列表"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        if path.suffix.lower() not in SUPPORTED_FORMATS:
            raise ValueError(f"不支持的格式: {path.suffix}（支持: {SUPPORTED_FORMATS}）")

        if path.suffix.lower() == ".pdf":
            docs = self._load_pdf(path)
        elif path.suffix.lower() == ".docx":
            docs = self._load_docx(path)
        else:
            docs = self._load_text(path)

        self.documents.extend(docs)
        return docs

    def load_directory(self, dir_path: str) -> List[Dict]:
        """加载目录下所有支持的文档"""
        d = Path(dir_path)
        if not d.exists():
            raise FileNotFoundError(f"目录不存在: {dir_path}")

        all_docs = []
        for f in sorted(d.iterdir()):
            if f.suffix.lower() in SUPPORTED_FORMATS:
                docs = self.load(str(f))
                all_docs.extend(docs)
                print(f"  加载: {f.name} ({len(docs)} 块)")

        print(f"共加载 {len(all_docs)} 个文档块")
        return all_docs

    def _load_pdf(self, path: Path) -> List[Dict]:
        from PyPDF2 import PdfReader
        reader = PdfReader(str(path))
        docs = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                docs.append({
                    "content": text.strip(),
                    "metadata": {
                        "source": path.name,
                        "page": i + 1,
                        "format": "pdf",
                        "modality": "text",
                        "doc_id": hashlib.md5(path.name.encode()).hexdigest()[:8],
                    },
                })

            images = self._extract_pdf_images(page, path.name, i + 1)
            for img_info in images:
                docs.append(img_info)

        return docs

    def _extract_pdf_images(self, page, source: str, page_num: int) -> List[Dict]:
        """从 PDF 页面提取图片"""
        images = []
        try:
            if "/XObject" not in page.get("/Resources", {}):
                return images

            x_object = page["/Resources"]["/XObject"].get_object()
            for obj_name in x_object:
                obj = x_object[obj_name]
                if obj.get("/Subtype") != "/Image":
                    continue

                width = obj.get("/Width", 0)
                height = obj.get("/Height", 0)
                if width < MIN_IMAGE_SIZE or height < MIN_IMAGE_SIZE:
                    continue

                img_dir = Path("./data/extracted_images")
                img_dir.mkdir(parents=True, exist_ok=True)
                img_name = f"{Path(source).stem}_p{page_num}_{obj_name.lstrip('/')}.png"
                img_path = img_dir / img_name

                try:
                    data = obj.get_data()
                    if obj.get("/Filter") == "/FlateDecode":
                        from PIL import Image
                        import zlib
                        try:
                            data = zlib.decompress(data)
                        except Exception:
                            pass
                        img = Image.frombytes("L", (width, height), data)
                        img.save(str(img_path))
                    elif obj.get("/ColorSpace") == "/DeviceRGB":
                        from PIL import Image
                        img = Image.frombytes("RGB", (width, height), data)
                        img.save(str(img_path))
                    else:
                        continue

                    images.append({
                        "content": f"[图像：{source} 第{page_num}页]",
                        "image_path": str(img_path),
                        "metadata": {
                            "source": source,
                            "page": page_num,
                            "format": "pdf",
                            "modality": "image",
                            "doc_id": hashlib.md5(source.encode()).hexdigest()[:8],
                            "image_name": img_name,
                            "image_size": [width, height],
                        },
                    })
                except Exception as e:
                    print(f"  图片提取失败 p{page_num} {obj_name}: {e}")
                    continue

        except Exception as e:
            print(f"  PDF图片提取异常 p{page_num}: {e}")

        return images

    def _load_docx(self, path: Path) -> List[Dict]:
        from docx import Document
        doc = Document(str(path))
        docs = []
        current_section = ""
        for para in doc.paragraphs:
            text = para.text.strip()
            if not text:
                continue
            if para.style and "Heading" in para.style.name:
                current_section = text
            section_tag = f" [{current_section}]" if current_section else ""
            docs.append({
                "content": text + section_tag,
                "metadata": {
                    "source": path.name,
                    "section": current_section,
                    "format": "docx",
                    "modality": "text",
                    "doc_id": hashlib.md5(path.name.encode()).hexdigest()[:8],
                },
            })

        img_dir = Path("./data/extracted_images")
        img_dir.mkdir(parents=True, exist_ok=True)
        for rel in doc.part.rels.values():
            if "image" in rel.reltype:
                try:
                    image_data = rel.target_part.blob
                    img_name = f"{path.stem}_img_{hashlib.md5(rel.target_ref.encode()).hexdigest()[:6]}.png"
                    img_path = img_dir / img_name
                    with open(img_path, "wb") as f:
                        f.write(image_data)
                    docs.append({
                        "content": f"[图像：{path.name}]",
                        "image_path": str(img_path),
                        "metadata": {
                            "source": path.name,
                            "format": "docx",
                            "modality": "image",
                            "doc_id": hashlib.md5(path.name.encode()).hexdigest()[:8],
                            "image_name": img_name,
                        },
                    })
                except Exception as e:
                    print(f"  DOCX图片提取失败: {e}")

        return docs

    def _load_text(self, path: Path) -> List[Dict]:
        text = path.read_text(encoding="utf-8", errors="ignore")
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        docs = []
        for i, para in enumerate(paragraphs):
            docs.append({
                "content": para,
                "metadata": {
                    "source": path.name,
                    "paragraph": i + 1,
                    "format": path.suffix.lower().strip("."),
                    "doc_id": hashlib.md5(path.name.encode()).hexdigest()[:8],
                },
            })
        return docs

    def get_stats(self) -> Dict:
        """获取已加载文档的统计信息"""
        sources = set(d["metadata"]["source"] for d in self.documents)
        formats = {}
        modalities = {"text": 0, "image": 0}
        for d in self.documents:
            fmt = d["metadata"]["format"]
            formats[fmt] = formats.get(fmt, 0) + 1
            mod = d["metadata"].get("modality", "text")
            modalities[mod] = modalities.get(mod, 0) + 1
        return {
            "total_chunks": len(self.documents),
            "total_documents": len(sources),
            "formats": formats,
            "modalities": modalities,
            "sources": list(sources),
        }
