import torch
import random
import logging
from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainingArguments
from transformers import AutoModelForCausalLM, DataCollatorForLanguageModeling, GPT2LMHeadModel
from transformers import AutoTokenizer

from defaults import *
from data import ArithmeticDataset
from data.templates import user_prompt as user_prompt_template


class Arithmetic:
	model: GPT2LMHeadModel
	def __init__(self) -> None:
		self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
		self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
		self.all_data = ArithmeticDataset(DATASET_SIZE, self.tokenizer)
		self.train_data, self.eval_data = train_test_split(self.all_data, test_size=TEST_SIZE, random_state=RANDOM_SEED)
		self.data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
		self.training_args = TrainingArguments(
			output_dir=OUTPUT_DIR,
			num_train_epochs=NUM_EPOCHS,
			per_device_train_batch_size=TRAIN_BATCH_SIZE,
			per_device_eval_batch_size=EVAL_BATCH_SIZE,
			gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
			learning_rate=LEARNING_RATE,
			lr_scheduler_type=LR_SCHEDULER_TYPE,
			weight_decay=WEIGHT_DECAY,
			warmup_steps=WARMUP_STEPS,
			logging_dir=LOGGING_DIR,
			logging_steps=LOGGING_STEPS,
			eval_strategy="steps",
			seed=RANDOM_SEED,
			report_to="tensorboard",
			bf16=True,
			bf16_full_eval=True,
		)
		self.trainer = Trainer(
			model=self.model,
			args=self.training_args,
			data_collator=self.data_collator,
			train_dataset=self.train_data,
			eval_dataset=self.eval_data
		)


	def train(self):
		logging.info("Start training...")
		self.trainer.train(resume_from_checkpoint=True)


	def sample_output(self):
		prompt = user_prompt_template.format(number_1=6794, number_2=8121)
		input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
		gen_tokens = self.model.generate(
			input_ids.to(self.model.device),
			do_sample=False,
			max_length=512
		)
		gen_text = self.tokenizer.batch_decode(gen_tokens)[0]
		return gen_text


def init():
	random.seed(RANDOM_SEED)	# Only set random seed for data generation
	logging.basicConfig(level=logging.INFO)
	if torch.cuda.is_available():
		logging.info("Training on CUDA...")
	else:
		logging.warning("CUDA not available. Training on CPU...")


if __name__ == "__main__":
	solver = Arithmetic()
	solver.train()
	print(solver.sample_output())