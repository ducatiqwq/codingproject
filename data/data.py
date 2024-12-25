import random
from typing import List

from .addition import get_data

class ArithmeticDataset:
	def __init__(self, siz: int, tokenizer) -> None:
		super().__init__()
		self.strs = self._generate_random_data(siz)
		self.encodings = self._tokenize_data(self.strs, tokenizer)

	def __getitem__(self, idx: int) -> dict:
		return {key: val[idx] for key, val in self.encodings.items()}

	def __len__(self) -> int:
		return len(self.strs)

	def _generate_random_data(self, siz: int) -> list:
		data = []
		for _ in range(siz):
			x = random.randint(0, 9999)
			y = random.randint(0, 9999)
			data.append(get_data(x, y))
		return data

	def _tokenize_data(self, data: List[str], tokenizer):
		tokenizer.pad_token = tokenizer.eos_token
		return tokenizer(data, return_tensors="pt", truncation=True, padding=True)