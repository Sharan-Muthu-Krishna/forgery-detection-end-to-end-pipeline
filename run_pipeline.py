from pipelines.ingestion_pipeline import ingestion_pipeline

if __name__ == "__main__":
    ingestion_pipeline(
        zip_path="data2.zip",
        extract_path="data2_extracted",
        dataset_root="data2",
        clean_path="clean_data",
        ela_path="ela_data_v1"
    )
