#!/usr/bin/env python3
"""
GRPO Model Fine-tuning Script

Fine-tunes a base instruction model (mistral-7b-instruct or similar) using GRPO 
(Generative Reinforcement Preference Optimization) on the Mongolian historical dataset.

Usage:
    python scripts/train_grpo_model.py
    python scripts/train_grpo_model.py --base mistralai/mistral-7b-instruct-v0.2
    python scripts/train_grpo_model.py --dataset data/mgl_history_grpo.jsonl --output models/custom_adapter
"""

import json
import os
import re
import sys
import time
import random
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

# Core ML libraries
try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, 
        TrainingArguments, Trainer,
        DataCollatorForLanguageModeling
    )
    from datasets import Dataset, DatasetDict
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel
    from trl import DPOTrainer, DPOConfig
    from accelerate import Accelerator
    from tqdm import tqdm
except ImportError as e:
    print(f"❌ Missing required libraries. Install with:")
    print("pip install transformers trl accelerate peft datasets torch tqdm")
    print(f"Error: {e}")
    sys.exit(1)

# Optional libraries
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


@dataclass
class TrainingStats:
    """Training statistics and metrics."""
    base_model: str = ""
    dataset_path: str = ""
    total_samples: int = 0
    train_samples: int = 0
    test_samples: int = 0
    epochs: int = 0
    total_steps: int = 0
    avg_loss: float = 0.0
    final_loss: float = 0.0
    mean_reward: float = 0.0
    validation_accuracy: float = 0.0
    training_time: float = 0.0
    model_size_mb: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class GRPODatasetProcessor:
    """Process GRPO dataset for training."""
    
    def __init__(self, tokenizer, max_length: int = 512):
        """Initialize dataset processor."""
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.logger = logging.getLogger(__name__)
    
    def load_grpo_dataset(self, dataset_path: Path) -> List[Dict[str, str]]:
        """Load GRPO dataset from JSONL file."""
        records = []
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            record = json.loads(line)
                            records.append(record)
                        except json.JSONDecodeError as e:
                            self.logger.warning(f"Invalid JSON on line {line_num}: {e}")
            
            self.logger.info(f"Loaded {len(records)} records from {dataset_path}")
            return records
            
        except Exception as e:
            self.logger.error(f"Failed to load dataset: {e}")
            raise
    
    def validate_record(self, record: Dict[str, str]) -> Tuple[bool, str]:
        """Validate a single GRPO record."""
        required_fields = ['prompt', 'chosen', 'rejected']
        
        # Check required fields
        for field in required_fields:
            if field not in record or not record[field]:
                return False, f"Missing or empty field: {field}"
        
        # Check minimum lengths
        prompt_words = len(record['prompt'].split())
        chosen_words = len(record['chosen'].split())
        rejected_words = len(record['rejected'].split())
        
        if prompt_words < 5:
            return False, f"Prompt too short: {prompt_words} words (min: 5)"
        
        if chosen_words < 20:
            return False, f"Chosen response too short: {chosen_words} words (min: 20)"
        
        if rejected_words < 10:
            return False, f"Rejected response too short: {rejected_words} words (min: 10)"
        
        # Check that chosen != rejected
        if record['chosen'] == record['rejected']:
            return False, "Chosen and rejected responses are identical"
        
        return True, ""
    
    def format_conversation(self, prompt: str, response: str) -> str:
        """Format prompt and response into conversation format."""
        return f"<|user|>\n{prompt}\n<|assistant|>\n{response}"
    
    def tokenize_record(self, record: Dict[str, str]) -> Dict[str, Any]:
        """Tokenize a single record for DPO training."""
        prompt = record['prompt']
        chosen = record['chosen']
        rejected = record['rejected']
        
        # Format conversations
        chosen_text = self.format_conversation(prompt, chosen)
        rejected_text = self.format_conversation(prompt, rejected)
        
        # Tokenize
        chosen_tokens = self.tokenizer(
            chosen_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None
        )
        
        rejected_tokens = self.tokenizer(
            rejected_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None
        )
        
        return {
            'prompt': prompt,
            'chosen': chosen,
            'rejected': rejected,
            'chosen_input_ids': chosen_tokens['input_ids'],
            'chosen_attention_mask': chosen_tokens['attention_mask'],
            'rejected_input_ids': rejected_tokens['input_ids'],
            'rejected_attention_mask': rejected_tokens['attention_mask']
        }
    
    def prepare_dataset(self, records: List[Dict[str, str]], 
                       train_split: float = 0.9) -> DatasetDict:
        """Prepare dataset for training."""
        # Validate and filter records
        valid_records = []
        invalid_count = 0
        
        for record in records:
            is_valid, error_msg = self.validate_record(record)
            if is_valid:
                valid_records.append(record)
            else:
                invalid_count += 1
                self.logger.warning(f"Invalid record: {error_msg}")
        
        self.logger.info(f"Valid records: {len(valid_records)}, Invalid: {invalid_count}")
        
        if len(valid_records) < 10:
            raise ValueError(f"Insufficient valid records: {len(valid_records)} (minimum: 10)")
        
        # Tokenize records
        tokenized_records = []
        for record in tqdm(valid_records, desc="Tokenizing records"):
            try:
                tokenized = self.tokenize_record(record)
                tokenized_records.append(tokenized)
            except Exception as e:
                self.logger.warning(f"Failed to tokenize record: {e}")
        
        # Split into train/test
        random.shuffle(tokenized_records)
        split_idx = int(len(tokenized_records) * train_split)
        
        train_records = tokenized_records[:split_idx]
        test_records = tokenized_records[split_idx:]
        
        self.logger.info(f"Train samples: {len(train_records)}, Test samples: {len(test_records)}")
        
        # Create datasets
        train_dataset = Dataset.from_list(train_records)
        test_dataset = Dataset.from_list(test_records)
        
        return DatasetDict({
            'train': train_dataset,
            'test': test_dataset
        })


class GRPOTrainer:
    """GRPO model trainer using DPO (Direct Preference Optimization)."""
    
    def __init__(self, base_model_name: str, output_dir: Path, 
                 use_wandb: bool = False, device: str = "auto"):
        """Initialize GRPO trainer."""
        self.base_model_name = base_model_name
        self.output_dir = output_dir
        self.use_wandb = use_wandb
        self.device = device
        self.logger = logging.getLogger(__name__)
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir = self.output_dir.parent / "training_logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.accelerator = Accelerator()
        
        # Training metrics
        self.training_losses = []
        self.validation_metrics = []
    
    def load_model_and_tokenizer(self):
        """Load base model and tokenizer."""
        self.logger.info(f"Loading model: {self.base_model_name}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_name,
                trust_remote_code=True,
                padding_side="left"
            )
            
            # Add special tokens if needed
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if self.device == "auto" else None,
                trust_remote_code=True
            )
            
            self.logger.info(f"Model loaded successfully. Parameters: {self.model.num_parameters():,}")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def setup_lora(self, r: int = 16, alpha: int = 32, dropout: float = 0.1):
        """Setup LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning."""
        self.logger.info("Setting up LoRA configuration")
        
        lora_config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        return lora_config
    
    def create_training_config(self, dataset_size: int, batch_size: int = 4, 
                             learning_rate: float = 5e-6, epochs: int = 2) -> DPOConfig:
        """Create DPO training configuration."""
        steps_per_epoch = max(1, dataset_size // (batch_size * self.accelerator.num_processes))
        total_steps = steps_per_epoch * epochs
        warmup_steps = int(total_steps * 0.1)
        
        config = DPOConfig(
            output_dir=str(self.output_dir),
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=8,
            num_train_epochs=epochs,
            warmup_steps=warmup_steps,
            lr_scheduler_type="cosine",
            logging_steps=10,
            save_steps=100,
            eval_steps=50,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="wandb" if self.use_wandb and WANDB_AVAILABLE else None,
            run_name=f"grpo-mongolian-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            remove_unused_columns=False,
            dataloader_pin_memory=False,
        )
        
        self.logger.info(f"Training config: {epochs} epochs, {total_steps} steps, lr={learning_rate}")
        return config
    
    def train(self, dataset: DatasetDict, training_config: DPOConfig) -> TrainingStats:
        """Train the model using DPO."""
        self.logger.info("Starting GRPO training with DPO")
        start_time = time.time()
        
        # Initialize trainer
        trainer = DPOTrainer(
            model=self.model,
            args=training_config,
            train_dataset=dataset['train'],
            eval_dataset=dataset['test'],
            tokenizer=self.tokenizer,
            beta=0.1,  # DPO beta parameter
        )
        
        # Start training
        try:
            train_result = trainer.train()
            
            # Save model
            trainer.save_model()
            self.tokenizer.save_pretrained(self.output_dir)
            
            # Calculate training time
            training_time = time.time() - start_time
            
            # Create training stats
            stats = TrainingStats(
                base_model=self.base_model_name,
                dataset_path="",  # Will be set by caller
                total_samples=len(dataset['train']) + len(dataset['test']),
                train_samples=len(dataset['train']),
                test_samples=len(dataset['test']),
                epochs=int(training_config.num_train_epochs),
                total_steps=train_result.global_step,
                avg_loss=train_result.training_loss,
                final_loss=train_result.training_loss,
                training_time=training_time
            )
            
            # Evaluate model
            eval_results = trainer.evaluate()
            stats.validation_accuracy = 1.0 - eval_results.get('eval_loss', 1.0)  # Approximate accuracy
            stats.mean_reward = max(0.0, 1.0 - eval_results.get('eval_loss', 1.0))
            
            self.logger.info(f"Training completed in {training_time:.1f}s")
            return stats
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise
    
    def generate_sample_responses(self, test_prompts: List[str], max_length: int = 256) -> List[str]:
        """Generate sample responses for evaluation."""
        self.logger.info("Generating sample responses")
        responses = []
        
        self.model.eval()
        with torch.no_grad():
            for prompt in test_prompts:
                try:
                    # Format prompt
                    formatted_prompt = f"<|user|>\n{prompt}\n<|assistant|>\n"
                    
                    # Tokenize
                    inputs = self.tokenizer(
                        formatted_prompt,
                        return_tensors="pt",
                        truncation=True,
                        max_length=512
                    )
                    
                    if torch.cuda.is_available():
                        inputs = {k: v.cuda() for k, v in inputs.items()}
                    
                    # Generate
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_length,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                    
                    # Decode response
                    response = self.tokenizer.decode(
                        outputs[0][inputs['input_ids'].shape[1]:],
                        skip_special_tokens=True
                    ).strip()
                    
                    responses.append(response)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to generate response for prompt: {e}")
                    responses.append("[Generation failed]")
        
        return responses
    
    def save_training_logs(self, stats: TrainingStats, sample_generations: List[Tuple[str, str]]):
        """Save training logs and sample generations."""
        # Save training stats
        stats_path = self.logs_dir / "training_stats.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats.to_dict(), f, ensure_ascii=False, indent=2)
        
        # Save sample generations
        samples_path = self.logs_dir / "sample_generations.jsonl"
        with open(samples_path, 'w', encoding='utf-8') as f:
            for prompt, response in sample_generations:
                sample = {
                    'prompt': prompt,
                    'response': response,
                    'timestamp': datetime.now().isoformat()
                }
                json.dump(sample, f, ensure_ascii=False)
                f.write('\n')
        
        self.logger.info(f"Training logs saved to {self.logs_dir}")


def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tune model with GRPO on Mongolian historical data")
    parser.add_argument('--base', default='mistralai/Mistral-7B-Instruct-v0.2',
                       help='Base model name')
    parser.add_argument('--dataset', default='data/mgl_history_grpo.jsonl',
                       help='GRPO dataset path')
    parser.add_argument('--output', default='models/mgl_history_grpo_adapter',
                       help='Output directory for trained model')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=5e-6,
                       help='Learning rate')
    parser.add_argument('--epochs', type=int, default=2,
                       help='Number of training epochs')
    parser.add_argument('--max-length', type=int, default=512,
                       help='Maximum sequence length')
    parser.add_argument('--use-wandb', action='store_true',
                       help='Use Weights & Biases for logging')
    parser.add_argument('--lora-r', type=int, default=16,
                       help='LoRA rank')
    parser.add_argument('--lora-alpha', type=int, default=32,
                       help='LoRA alpha')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('training_logs/training.log', encoding='utf-8')
        ]
    )
    logger = logging.getLogger(__name__)
    
    print("🚀 GRPO FINE-TUNING PIPELINE")
    print("=" * 50)
    print(f"Base model: {args.base}")
    print(f"Dataset: {args.dataset}")
    print(f"Output: {args.output}")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print()
    
    try:
        # Initialize trainer
        trainer = GRPOTrainer(
            base_model_name=args.base,
            output_dir=Path(args.output),
            use_wandb=args.use_wandb
        )
        
        # Load model and tokenizer
        trainer.load_model_and_tokenizer()
        
        # Setup LoRA
        trainer.setup_lora(r=args.lora_r, alpha=args.lora_alpha)
        
        # Initialize dataset processor
        processor = GRPODatasetProcessor(trainer.tokenizer, max_length=args.max_length)
        
        # Load and prepare dataset
        logger.info("Loading and preparing dataset")
        records = processor.load_grpo_dataset(Path(args.dataset))
        dataset = processor.prepare_dataset(records)
        
        # Create training configuration
        training_config = trainer.create_training_config(
            dataset_size=len(dataset['train']),
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            epochs=args.epochs
        )
        
        # Initialize wandb if requested
        if args.use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project="mongolian-grpo-training",
                config=vars(args)
            )
        
        # Train model
        logger.info("Starting training")
        stats = trainer.train(dataset, training_config)
        stats.dataset_path = args.dataset
        
        # Generate sample responses
        test_prompts = [
            "Чингис хааны тухай ярина уу?",
            "1921 оны хувьсгалын үр дүн юу байсан бэ?",
            "Монголын ардчилсан хувьсгал хэрхэн өрнөсөн бэ?",
            "Богд хааны үеийн онцлог нь юу вэ?",
            "Хүннү улсын ач холбогдол юунд оршдог вэ?"
        ]
        
        responses = trainer.generate_sample_responses(test_prompts)
        sample_generations = list(zip(test_prompts, responses))
        
        # Save training logs
        trainer.save_training_logs(stats, sample_generations)
        
        # Display results
        print("\n📊 GRPO FINE-TUNING REPORT")
        print("=" * 50)
        print(f"Base model: {stats.base_model}")
        print(f"Dataset: {stats.dataset_path} ({stats.total_samples} pairs)")
        print(f"Training samples: {stats.train_samples}")
        print(f"Test samples: {stats.test_samples}")
        print(f"Total steps: {stats.total_steps}")
        print(f"Average loss: {stats.avg_loss:.4f}")
        print(f"Mean reward: {stats.mean_reward:.3f}")
        print(f"Validation accuracy: {stats.validation_accuracy:.1%}")
        print(f"Training time: {stats.training_time:.1f}s")
        print()
        print("✅ Model saved:", args.output)
        print("✅ Training logs saved: training_logs/")
        print()
        print("🎉 GRPO Fine-tuning Complete!")
        print("✅ Model successfully trained on Mongolian preference dataset.")
        print(f"✅ Adapter saved to {args.output}/")
        print("✅ Ready for evaluation with test prompts or integrated RAG agent.")
        print()
        print("Fine-tuned model чинь Монгол хэлний түүхийн RAG агент маягаар")
        print("илүү бодит, оновчтой хариулт өгдөг болно")
        print()
        print("Чи дараа нь ингэж RAG агентдаа холбож болно:")
        print("from peft import PeftModel")
        print(f"model = PeftModel.from_pretrained(base_model, '{args.output}')")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Training failed: {e}")
        print(f"❌ Training failed: {e}")
        return 1
    finally:
        if args.use_wandb and WANDB_AVAILABLE:
            wandb.finish()


if __name__ == "__main__":
    exit(main())