import random, os, json 
import numpy as np
from datasets import load_dataset
import torch
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, 
    DataCollatorForSeq2Seq, EarlyStoppingCallback
)
from evaluate import load as load_metric
from utils import tool

def set_seed(seed):
    random.seed(seed)           # python
    np.random.seed(seed)        # numpy
    torch.manual_seed(seed)     # init matric weight/dropout

    if torch.cuda.is_available():   # init matric weight/dropout
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 

    torch.backends.cudnn.deterministic = True      # CNN algo
    torch.backends.cudnn.benchmark = False

def run(cfg):
    # ---------------------
    # 1. set seed
    # ---------------------
    if cfg["seed"]:
        set_seed(cfg["seed"])
        print(f"🌱 Random seed set to {cfg['seed']}")

    # ---------------------
    # 2. load cleaned data
    # ---------------------
    root_dir = tool.get_root_dir(cfg)
    train_file = os.path.join(root_dir,cfg["data"]["train_set"] )
    val_file = os.path.join(root_dir, cfg["data"]["val_set"])

    train_data = load_dataset("json", data_files=train_file)["train"]
    val_data = load_dataset("json", data_files=val_file)["train"]
    print(f"✅ Train samples: {len(train_data):,} | Val samples: {len(val_data):,}")

    # ---------------------
    # 3. initialize model and tokenizer
    # ---------------------
    model_cfg = cfg["model"]
    print(f"🤖 Loading base model: {model_cfg['base_model']}")
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["tokenizer"])

    model = AutoModelForSeq2SeqLM.from_pretrained(model_cfg["base_model"]).to("cuda" if torch.cuda.is_available() else "cpu")

    max_input = model_cfg["max_input_len"]
    max_output = model_cfg["max_output_len"]
    print(f"🧩 Tokenizer vocab size: {len(tokenizer)} | model max length: {max_input}")

    # ---------------------
    # 4. Tokenizer encoding
    # ---------------------
    print("🔧 Tokenizing datasets ...")
    def tokenize_function(batch):
        inputs = batch["review"]
        targets = batch["summary"]

        # tokenizer return "input_ids"、"attention_mask"
        model_inputs = tokenizer(
            inputs,
            max_length=max_input,
            truncation=True,
            padding="max_length"
        )

        with tokenizer.as_target_tokenizer():
            labels= tokenizer(
                targets,
                max_length=max_output,
                truncation=True,
                padding="max_length"
            )["input_ids"]
        
        model_inputs["labels"] = labels
        return model_inputs

    # after tokenizer : { "input_ids": [...], "attention_mask": [...], "labels": [...] }
    tokenized_train = train_data.map(tokenize_function, batched=True, remove_columns=train_data.column_names)
    tokenized_val = val_data.map(tokenize_function, batched=True, remove_columns=val_data.column_names)

    # ---------------------
    # 5. Data collator  - pack data to tensor
    # ---------------------
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # ---------------------
    # 6. 评估指标 (ROUGE)
    # ---------------------
    train_cfg = cfg["train"]
    metric = load_metric("rouge")   #ROUGE 衡量的是「生成摘要与参考摘要」之间的重叠程度

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True) # reverse to text. skip <pad>、<s>、</s>
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result = {k: round(v.mid.fmeasure * 100, 2) for k, v in result.items()}
        return result
    
    # ---------------------
    # 7. 训练参数设置
    # ---------------------
    in_colab = "COLAB_GPU" in os.environ or "COLAB_RELEASE_TAG" in os.environ

    if in_colab:
        output_dir = cfg["output"]["dir_colab"]
    else:
        output_dir = cfg["output"]["dir_local"]   # 相对路径即可
    os.makedirs(output_dir, exist_ok=True)

    args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        learning_rate=train_cfg["learning_rate"],
        per_device_train_batch_size=train_cfg["batch_size"],
        num_train_epochs=train_cfg["epochs"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        eval_strategy=train_cfg["evaluation_strategy"],
        eval_steps=train_cfg["eval_steps"],
        save_steps=train_cfg["save_steps"],
        predict_with_generate=True,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=100,
        report_to=train_cfg.get("report_to", ["tensorboard"]),
        load_best_model_at_end=True,
        metric_for_best_model=train_cfg["early_stopping"]["metric"]
            if train_cfg.get("early_stopping", {}).get("enabled")
            else None,
        greater_is_better=train_cfg["early_stopping"].get("mode", "max") == "max",   
        resume_from_checkpoint=True,
        save_total_limit=2,
        fp16=True,
    )

    # ---------------------
    # 8. Trainer 设置
    # ---------------------
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # ---------------------
    # 9. Early Stopping
    # ---------------------
    if train_cfg.get("early_stopping", {}).get("enabled"):
        patience = train_cfg["early_stopping"]["patience"]
        trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=patience))
        print(f"⏸️ EarlyStopping enabled (patience={patience})")

    # ---------------------
    # 10. 开始训练
    # ---------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"💻 Training on {device.upper()}")

    print("🚀 Start SFT training ...")
    trainer.train()

    best_metrics = trainer.state.best_metric
    if best_metrics is not None:
        print(f"🏆 Best eval metric ({args.metric_for_best_model}): {best_metrics:.2f}")
    else:
        print("⚠️ No evaluation metric recorded (probably training was interrupted early).")

    # ---------------------
    # 11. save model
    # ---------------------
    print(f"💾 Saving best model to {output_dir}")
    trainer.save_model(output_dir)
    if not os.path.exists(os.path.join(output_dir, "tokenizer_config.json")):
        tokenizer.save_pretrained(output_dir)
        print("🧾 Tokenizer saved to output directory.")

    print("\n📂 Model artifacts saved under:")
    for f in os.listdir(output_dir):
        print("  ├─", f)

    # run evaluate for saving final_metrics.json
    try:
        final_metrics = trainer.evaluate()
        with open(os.path.join(output_dir, "final_metrics.json"), "w") as f:
            json.dump(final_metrics, f, indent=2)
        print("📊 Final evaluation metrics saved to final_metrics.json")
    except Exception as e:
        print(f"⚠️ Evaluation skipped or failed: {e}")

    print("✅ SFT training complete!")

    return {
        "best_metric": best_metrics,
        "final_metrics":  locals().get("final_metrics", {}),
        "output_dir": output_dir
    }