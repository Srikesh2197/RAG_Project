"""Download MultiHop-RAG at the pinned revision and build the processed files."""

import argparse

from huggingface_hub import hf_hub_download

from ragbasics.config import DataConfig, load_config
from ragbasics.eval.dataset import prepare


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config, DataConfig)

    for filename in (cfg.corpus_file, cfg.questions_file):
        hf_hub_download(
            cfg.repo_id,
            filename,
            repo_type="dataset",
            revision=cfg.revision,
            local_dir=cfg.raw_dir,
        )

    manifest = prepare(cfg)
    mapping = manifest["mapping"]
    print(f"documents: {manifest['documents']}")
    print(f"questions: {manifest['questions']}")
    print(
        f"evidence mapped to spans: {mapping['mapped']}/{mapping['evidence_total']} "
        f"({mapping['rate']:.2%}); exact {mapping['matched_exact']}, "
        f"after whitespace collapse {mapping['matched_after_whitespace']}, "
        f"article missing {mapping['article_not_in_corpus']}, "
        f"sentence missing {mapping['sentence_not_in_article']}"
    )
    check = manifest["split_check"]
    print(
        f"gold spans shared by dev and test: {check['gold_spans_shared_by_dev_and_test']}; "
        f"largest cluster share: dev {check['largest_cluster_share']['dev']:.1%}, "
        f"test {check['largest_cluster_share']['test']:.1%}"
    )
    print(f"wrote {cfg.processed_dir}/")


if __name__ == "__main__":
    main()
