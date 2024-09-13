from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling

def fine_tune_model(model_name, train_file, output_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Prepare dataset
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=train_file,
        block_size=128)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        save_steps=10_000,
        save_total_limit=2)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset)

    trainer.train()
    trainer.save_model(output_dir)
    
def fine_tune_language_model(model_name, train_file, output_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=train_file,
        block_size=128
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=10_000,
        save_total_limit=2
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset
    )

    trainer.train()
    trainer.save_model(output_dir)

    return model, tokenizer

# Usage
if __name__ == "__main__":
    fine_tune_model("EleutherAI/gpt-neo-125M", "code_training_data.txt", "./fine_tuned_model")
    model, tokenizer = fine_tune_language_model("EleutherAI/gpt-neo-125M", "domain_data.txt", "./fine_tuned_model")
