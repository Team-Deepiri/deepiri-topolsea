use clap::Parser;
use dv_bench::{run_ann_bench, try_load_sift_or_synthetic};
use dv_types::{DistanceMetric, IndexKind};
use std::path::PathBuf;
use std::str::FromStr;

#[derive(Parser, Debug)]
#[command(name = "topolsea-ann-bench")]
#[command(about = "ANN-Benchmarks-style recall/QPS runner (B9)")]
struct Args {
    /// Directory with sift_base.fvecs / sift_query.fvecs / sift_groundtruth.ivecs
    #[arg(long)]
    data_dir: Option<PathBuf>,

    #[arg(long, default_value = "hnsw")]
    index: String,

    #[arg(long, default_value = "l2")]
    metric: String,

    #[arg(long, default_value_t = 10)]
    top_k: usize,
}

fn main() {
    let args = Args::parse();
    let (name, ds) = try_load_sift_or_synthetic(args.data_dir);
    let kind = match args.index.to_lowercase().as_str() {
        "flat" => IndexKind::Flat,
        "zcolumn" => IndexKind::ZColumn,
        "ivf" => IndexKind::Ivf,
        _ => IndexKind::Hnsw,
    };
    let metric = DistanceMetric::from_str(&args.metric).unwrap_or(DistanceMetric::L2);
    let report = run_ann_bench(&name, &ds, kind, metric, args.top_k);
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
}
