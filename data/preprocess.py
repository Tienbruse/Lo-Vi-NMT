import os
import sentencepiece as spm
from mosestokenizer import MosesTokenizer
from sacremoses import MosesTruecaser, MosesPunctNormalizer
from typing import List
from pathlib import Path

class NMTPreprocessor:
    """Lớp xử lý tiền xử lý dữ liệu cho dịch máy đa ngôn ngữ"""
    
    def __init__(
        self,
        data_dir: str,
        output_dir: str,
        src_langs: List[str],
        tgt_lang: str,
        vocab_size: int = 32000,
        datasets: List[str] = ["train", "dev", "test"]
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.src_langs = src_langs
        self.tgt_lang = tgt_lang
        self.vocab_size = vocab_size
        self.datasets = datasets
        
        # Tạo thư mục đầu ra
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Đường dẫn mô hình
        self.sp_model_prefix = self.output_dir / "spm_model"
        self.truecase_model = self.output_dir / f"truecase_model.{tgt_lang}"

    def train_sentencepiece(self, dataset: str = "train") -> None:
        """Huấn luyện mô hình SentencePiece trên tập dữ liệu nguồn"""
        input_files = [
            str(self.data_dir / f"{dataset}.{lang}")
            for lang in self.src_langs
            if (self.data_dir / f"{dataset}.{lang}").exists()
        ]
        
        if not input_files:
            raise FileNotFoundError(f"No source files found for {dataset} dataset!")
        
        spm.SentencePieceTrainer.train(
            input=",".join(input_files),
            model_prefix=str(self.sp_model_prefix),
            vocab_size=self.vocab_size,
            character_coverage=0.9995,
            model_type="bpe",
            max_sentence_length=10000
        )
        print(f"Trained SentencePiece model: {self.sp_model_prefix}.model")

    def apply_sentencepiece(self, input_file: Path, output_file: Path) -> None:
        """Áp dụng SentencePiece để tokenize và mã hóa BPE"""
        if not input_file.exists():
            print(f"Warning: File {input_file} not found!")
            return
        
        sp = spm.SentencePieceProcessor()
        sp.load(str(self.sp_model_prefix) + ".model")
        
        with open(input_file, "r", encoding="utf-8") as fin, \
             open(output_file, "w", encoding="utf-8") as fout:
            for line in fin:
                tokenized = sp.encode(line.strip(), out_type=str)
                fout.write(" ".join(tokenized) + "\n")
        print(f"Processed {input_file} -> {output_file}")

    def train_truecaser(self, dataset: str = "train") -> None:
        """Huấn luyện mô hình true-casing cho ngôn ngữ đích"""
        input_file = self.data_dir / f"{dataset}.{self.tgt_lang}"
        if not input_file.exists():
            raise FileNotFoundError(f"File {input_file} not found!")
        
        truecaser = MosesTruecaser()
        truecaser.train([str(input_file)], str(self.truecase_model))
        print(f"Trained true-caser model: {self.truecase_model}")

    def apply_moses_vi(self, input_file: Path, output_file: Path) -> None:
        """Tokenize và true-case ngôn ngữ đích (tiếng Việt)"""
        if not input_file.exists():
            print(f"Warning: File {input_file} not found!")
            return
        
        if not self.truecase_model.exists():
            raise FileNotFoundError(f"Truecase model {self.truecase_model} not found!")
        
        tokenizer = MosesTokenizer(self.tgt_lang)
        truecaser = MosesTruecaser(str(self.truecase_model))
        punct_normalizer = MosesPunctNormalizer(self.tgt_lang)
        
        with open(input_file, "r", encoding="utf-8") as fin, \
             open(output_file, "w", encoding="utf-8") as fout:
            for line in fin:
                # Chuẩn hóa dấu câu
                normalized = punct_normalizer.normalize(line.strip())
                if not normalized:
                    fout.write("\n")
                    continue
                # True-case chuỗi văn bản
                truecased = truecaser.truecase(normalized)
                # Nếu truecased là danh sách token, nối thành chuỗi
                if isinstance(truecased, list):
                    truecased_text = " ".join(truecased)
                else:
                    truecased_text = truecased
                # Tokenize chuỗi đã true-case
                tokens = tokenizer(truecased_text)
                fout.write(" ".join(tokens) + "\n")
        
        tokenizer.close()
        print(f"Processed {input_file} -> {output_file}")

    def preprocess_dataset(self, dataset: str) -> None:
        """Xử lý một tập dữ liệu (train, dev, hoặc test)"""
        print(f"\nProcessing {dataset} dataset...")
        
        # Xử lý ngôn ngữ nguồn
        for lang in self.src_langs:
            input_file = self.data_dir / f"{dataset}.{lang}"
            output_file = self.output_dir / f"{dataset}.{lang}"
            self.apply_sentencepiece(input_file, output_file)
        
        # Xử lý ngôn ngữ đích (tiếng Việt)
        input_file = self.data_dir / f"{dataset}.{self.tgt_lang}"
        output_file = self.output_dir / f"{dataset}.{self.tgt_lang}"
        self.apply_moses_vi(input_file, output_file)

    def preprocess_all(self) -> None:
        """Xử lý tất cả các tập dữ liệu"""
        # Huấn luyện mô hình chỉ trên tập train
        print("Training SentencePiece and true-caser on train dataset...")
        self.train_sentencepiece(dataset="train")
        self.train_truecaser(dataset="train")
        
        # Xử lý từng tập dữ liệu
        for dataset in self.datasets:
            self.preprocess_dataset(dataset)
        
        print("\nPreprocessing completed!")

def main():
    # Cấu hình
    config = {
        "data_dir": "data/merged",
        "output_dir": "processed_data",
        "src_langs": ["lo"],
        "tgt_lang": "vi",
        "vocab_size": 16000,
        "datasets": ["train", "dev", "test"]
    }
    
    # Khởi tạo và chạy tiền xử lý
    preprocessor = NMTPreprocessor(**config)
    preprocessor.preprocess_all()

if __name__ == "__main__":
    main()