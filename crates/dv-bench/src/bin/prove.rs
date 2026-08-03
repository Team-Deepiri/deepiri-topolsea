use dv_bench::{run, ProveConfig};
use std::env;

fn main() {
    let mut config = ProveConfig::default();
    let args: Vec<String> = env::args().collect();
    if args.iter().any(|a| a == "--million") {
        config.scales.push(1_000_000);
    }
    if let Some(pos) = args.iter().position(|a| a == "--scale") {
        if let Some(s) = args.get(pos + 1).and_then(|v| v.parse().ok()) {
            config.scales = vec![s];
        }
    }

    let report = run(config);
    println!("{}", serde_json::to_string_pretty(&report).expect("json"));
}
