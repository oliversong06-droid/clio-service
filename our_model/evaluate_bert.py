#!/usr/bin/env python3
"""Evaluate the fine-tuned BERT emotion model and export the report to files."""

import argparse
import numbers
import os
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer

TARGET_LABELS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
LABEL_NORMALIZATION = {
    "anger": "anger",
    "angry": "anger",
    "annoyance": "anger",
    "annoyed": "anger",
    "rage": "anger",
    "furious": "anger",
    "mad": "anger",
    "disgust": "disgust",
    "disgusted": "disgust",
    "hate": "disgust",
    "hatred": "disgust",
    "aversion": "disgust",
    "gross": "disgust",
    "fear": "fear",
    "afraid": "fear",
    "worry": "fear",
    "worried": "fear",
    "anxiety": "fear",
    "panic": "fear",
    "scared": "fear",
    "joy": "joy",
    "happy": "joy",
    "happiness": "joy",
    "fun": "joy",
    "enthusiasm": "joy",
    "relief": "joy",
    "love": "joy",
    "delight": "joy",
    "pleasure": "joy",
    "sadness": "sadness",
    "sad": "sadness",
    "empty": "sadness",
    "boredom": "sadness",
    "lonely": "sadness",
    "grief": "sadness",
    "sorrow": "sadness",
    "surprise": "surprise",
    "surprised": "surprise",
    "shock": "surprise",
    "astonished": "surprise",
}

def clean_text(text: str) -> str:
    """Mirror the preprocessing that was used during training."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = "".join(ch if ch.isalpha() or ch.isspace() else " " for ch in text)
    return " ".join(text.split())


def normalize_label(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = str(value).strip().lower()
    return LABEL_NORMALIZATION.get(value, value if value in TARGET_LABELS else None)


def prepare_dataframe(
    df: pd.DataFrame,
    allowed_labels: Optional[Iterable[str]],
    max_rows: Optional[int],
) -> Tuple[List[str], List[str], List[str]]:
    working = df[["text", "label"]].dropna().copy()
    working["text"] = working["text"].apply(clean_text)
    working["label"] = working["label"].apply(normalize_label)
    working.dropna(subset=["text", "label"], inplace=True)

    if allowed_labels is not None:
        allowed = {label.lower() for label in allowed_labels}
        working = working[working["label"].isin(allowed)]

    if max_rows is not None and max_rows > 0 and len(working) > max_rows:
        working = working.sample(n=max_rows, random_state=42).reset_index(drop=True)

    if working.empty:
        raise RuntimeError("Dataset is empty after preprocessing.")

    texts = working["text"].tolist()
    labels = working["label"].tolist()
    class_names = sorted(working["label"].unique())
    return texts, labels, class_names


def load_csv_dataset(
    csv_path: str,
    allowed_labels: Optional[Iterable[str]],
    max_rows: Optional[int],
) -> Tuple[List[str], List[str], List[str]]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path, encoding="latin1")
    df = df.rename(columns={"Emotion": "label", "text": "text"})
    return prepare_dataframe(df, allowed_labels, max_rows)


def _decode_hf_label(value, label_names: Optional[List[str]]):
    if value is None:
        return None
    if isinstance(value, list):
        if not value:
            return None
        value = value[0]
    if label_names is not None and isinstance(value, numbers.Integral):
        idx = int(value)
        if 0 <= idx < len(label_names):
            return label_names[idx]
    return value


def load_hf_dataset(
    dataset_name: str,
    subset: Optional[str],
    split: str,
    text_field: str,
    label_field: str,
    allowed_labels: Optional[Iterable[str]],
    max_rows: Optional[int],
) -> Tuple[List[str], List[str], List[str]]:
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "datasets 라이브러리가 필요합니다. `pip install datasets`로 설치하세요."
        ) from exc

    dataset = load_dataset(dataset_name, name=subset, split=split)

    label_feature = dataset.features.get(label_field)
    label_names: Optional[List[str]] = None
    if label_feature is not None:
        if hasattr(label_feature, "names"):
            label_names = list(label_feature.names)
        elif hasattr(label_feature, "feature") and hasattr(label_feature.feature, "names"):
            label_names = list(label_feature.feature.names)

    records = []
    for row in dataset:
        text_value = row[text_field]
        label_value = _decode_hf_label(row[label_field], label_names)
        records.append({"text": text_value, "label": label_value})

    df = pd.DataFrame(records)
    return prepare_dataframe(df, allowed_labels, max_rows)


def load_dataset_from_args(
    args,
    allowed_labels: Optional[Iterable[str]],
) -> Tuple[List[str], List[str], List[str]]:
    if args.data_source == "hf":
        if not args.hf_dataset:
            raise ValueError("`--hf-dataset` 값을 지정해야 합니다.")
        return load_hf_dataset(
            dataset_name=args.hf_dataset,
            subset=args.hf_subset,
            split=args.hf_split,
            text_field=args.hf_text_field,
            label_field=args.hf_label_field,
            allowed_labels=allowed_labels,
            max_rows=args.max_rows,
        )
    return load_csv_dataset(args.dataset, allowed_labels, args.max_rows)


def load_model(model_dir: str):
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    return tokenizer, model


def predict_labels(
    texts: List[str],
    tokenizer,
    model,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> List[str]:
    id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
    preds = []

    model.to(device)
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            outputs = model(**encoded)
            batch_preds = torch.argmax(outputs.logits, dim=1).cpu().tolist()
            preds.extend(id2label[idx] for idx in batch_preds)
    return preds


def save_report(
    report_path: str,
    accuracy: float,
    class_report: str,
    cm,
    class_names: List[str],
):
    with open(report_path, "w", encoding="utf-8") as fp:
        fp.write("BERT Emotion Model Evaluation\n")
        fp.write("--------------------------------\n")
        fp.write(f"Accuracy: {accuracy * 100:.2f}%\n\n")
        fp.write("[Classification Report]\n")
        fp.write(class_report)
        fp.write("\n\n[Confusion Matrix]\n")
        header = " " * 12 + " ".join(f"{label:>10}" for label in class_names)
        fp.write(header + "\n")
        for label, row in zip(class_names, cm):
            fp.write(f"{label:>10} " + " ".join(f"{count:10d}" for count in row) + "\n")


def save_confusion_matrix_plot(cm, class_names: List[str], output_path: str):
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("BERT Emotion Model Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def evaluate(args):
    tokenizer, model = load_model(args.model_dir)
    model_labels = sorted({label.lower() for label in model.config.id2label.values()})

    desired_labels = [label.lower() for label in (args.labels or TARGET_LABELS)]
    missing_in_model = [label for label in desired_labels if label not in model_labels]
    if missing_in_model:
        raise RuntimeError(
            "모델이 다음 감정 라벨을 지원하지 않습니다: "
            + ", ".join(missing_in_model)
            + ". BERT를 다시 파인튜닝하세요."
        )

    texts, labels, available_classes = load_dataset_from_args(args, desired_labels)
    print(
        f"📊 평가용 데이터 로드 완료: {len(labels)} samples / labels = "
        + ", ".join(sorted(set(labels)))
    )

    if len(set(labels)) < 2:
        raise RuntimeError("평가하려면 최소 두 개의 감정 라벨이 필요합니다.")

    _, X_test, _, y_test = train_test_split(
        texts,
        labels,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=labels,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y_pred = predict_labels(
        X_test,
        tokenizer,
        model,
        device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    class_names = desired_labels
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test,
        y_pred,
        labels=class_names,
        target_names=class_names,
        digits=4,
    )
    cm = confusion_matrix(y_test, y_pred, labels=class_names)

    os.makedirs(args.output_dir, exist_ok=True)
    text_report_path = os.path.join(args.output_dir, "bert_evaluation_report.txt")
    heatmap_path = os.path.join(args.output_dir, "bert_confusion_matrix.png")

    save_report(text_report_path, accuracy, report, cm, class_names)
    save_confusion_matrix_plot(cm, class_names, heatmap_path)

    print(f"Saved evaluation report to: {text_report_path}")
    print(f"Saved confusion matrix figure to: {heatmap_path}")


def parse_args():
    default_root = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Evaluate the fine-tuned BERT model.")
    parser.add_argument(
        "--data-source",
        choices=["csv", "hf"],
        default="csv",
        help="평가용 데이터를 CSV에서 읽을지(HF 라이브러리) 선택합니다.",
    )
    parser.add_argument(
        "--dataset",
        default=os.path.join(default_root, "emotion_sentimen_dataset.csv"),
        help="CSV 데이터 경로 (`--data-source csv` 일 때 사용).",
    )
    parser.add_argument(
        "--hf-dataset",
        default=None,
        help="Hugging Face datasets 라이브러리에서 불러올 데이터셋 이름.",
    )
    parser.add_argument(
        "--hf-subset",
        default=None,
        help="데이터셋에 서브셋/이름 인자가 필요한 경우 지정 (예: tweet_eval의 emotion).",
    )
    parser.add_argument(
        "--hf-split",
        default="test",
        help="Hugging Face 데이터셋에서 사용할 split 이름 (예: train, validation, test).",
    )
    parser.add_argument(
        "--hf-text-field",
        default="text",
        help="Hugging Face 데이터셋의 텍스트 컬럼 이름.",
    )
    parser.add_argument(
        "--hf-label-field",
        default="label",
        help="Hugging Face 데이터셋의 라벨 컬럼 이름.",
    )
    parser.add_argument(
        "--model-dir",
        default=os.path.join(default_root, "bert_finetuned"),
        help="Directory that stores the fine-tuned BERT weights.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(default_root, "bert_eval_outputs"),
        help="Directory where the report and confusion matrix image will be saved.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of the dataset to use for testing.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Maximum token length for BERT inputs.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap on the number of dataset rows to load (random sample).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for the train/test split.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="평가에 사용할 라벨 순서 (기본값은 anger/disgust/fear/joy/sadness/surprise).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
