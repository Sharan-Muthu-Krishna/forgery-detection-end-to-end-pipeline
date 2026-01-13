from pipelines.train_eval_deploy_pipeline import train_eval_deploy_pipeline

if __name__ == "__main__":
    train_eval_deploy_pipeline(
        ela_data_path="ela_data_v1",
        base_model_path="forgery_ela_mobilenet_finetuned.keras",
        test_dir="test_data",
        clean_dir="test_clean",
        ela_dir="test_ela"
    )
