use clap::{Parser, Subcommand};
use dv_bench::ProveConfig;
use dv_query::{Database, ShardQueryServer, ShardServerConfig};
use dv_types::{DistanceMetric, IndexKind};
use std::path::PathBuf;
use std::str::FromStr;

#[derive(Parser)]
#[command(name = "topolsea", about = "Deepiri Topolsea vector database CLI")]
struct Cli {
    #[arg(long, default_value = "./data")]
    data_dir: PathBuf,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// List all collections
    List,
    /// Create a new collection
    Create {
        name: String,
        #[arg(long)]
        dimension: usize,
        #[arg(long, default_value = "cosine")]
        metric: String,
        #[arg(long, default_value = "hnsw")]
        index: String,
    },
    /// Delete a collection
    Delete { name: String },
    /// Show collection info
    Info { name: String },
    /// Search a collection (prints hits + Z-Column explain when applicable)
    Search {
        collection: String,
        #[arg(long, value_delimiter = ',')]
        vector: Vec<f32>,
        #[arg(long, default_value = "5")]
        top_k: usize,
        #[arg(long)]
        explain: bool,
    },
    /// Show recommended index plan for a workload
    Plan {
        #[arg(long, default_value = "10000")]
        size: usize,
        #[arg(long, default_value = "128")]
        dimension: usize,
        #[arg(long, default_value = "10")]
        top_k: usize,
    },
    /// Create a fractal-sharded logical collection (M4)
    ShardCreate {
        name: String,
        #[arg(long, default_value = "4")]
        shards: usize,
        #[arg(long)]
        dimension: usize,
        #[arg(long, default_value = "cosine")]
        metric: String,
        #[arg(long, default_value = "zcolumn")]
        index: String,
    },
    /// Batch search (multiple query vectors)
    BatchSearch {
        collection: String,
        #[arg(long)]
        vectors_file: PathBuf,
        #[arg(long, default_value = "5")]
        top_k: usize,
        #[arg(long)]
        sharded: bool,
    },
    /// Serve HTTP shard queries for a physical collection (cross-node fan-out)
    ShardServe {
        collection: String,
        #[arg(long, default_value = "127.0.0.1:7700")]
        bind: String,
    },
    /// Run commercial proof report (recall, QPS, footprint) — bench-only, no hot-path cost hooks
    Prove {
        #[arg(long)]
        million: bool,
        #[arg(long)]
        scale: Option<usize>,
        #[arg(long, default_value = "50")]
        queries: usize,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    let mut db = Database::open(&cli.data_dir)?;

    match cli.command {
        Commands::List => {
            for name in db.list_collections()? {
                println!("{name}");
            }
        }
        Commands::Create {
            name,
            dimension,
            metric,
            index,
        } => {
            let metric = DistanceMetric::from_str(&metric)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e))?;
            let index_kind = match index.to_lowercase().as_str() {
                "flat" => IndexKind::Flat,
                "zcolumn" => IndexKind::ZColumn,
                _ => IndexKind::Hnsw,
            };
            let mut config = dv_types::CollectionConfig::new(name.clone(), dimension, metric);
            config.index_kind = index_kind;
            if index_kind == IndexKind::Flat {
                config = config.with_flat_index();
            } else if index_kind == IndexKind::ZColumn {
                config = config.with_zcolumn_index();
            }
            db.create_collection(config)?;
            println!(
                "created collection '{name}' (dim={dimension}, metric={metric}, index={index})"
            );
        }
        Commands::Delete { name } => {
            db.delete_collection(&name)?;
            println!("deleted collection '{name}'");
        }
        Commands::Info { name } => {
            let col = db.get_collection(&name)?;
            println!("name: {}", col.name());
            println!("dimension: {}", col.config().dimension);
            println!("metric: {}", col.config().metric);
            println!("index: {:?}", col.config().index_kind);
            println!("vectors: {}", col.len());
            if let Some(stats) = col.zcolumn_stats() {
                println!("zcolumn_stats: {stats}");
            }
        }
        Commands::Search {
            collection,
            vector,
            top_k,
            explain,
        } => {
            let col = db.get_collection(&collection)?;
            if vector.len() != col.config().dimension {
                return Err(format!(
                    "vector dimension {} != collection dimension {}",
                    vector.len(),
                    col.config().dimension
                )
                .into());
            }
            if explain {
                let (results, ex) = col.query_explain(&vector, top_k, None, 64)?;
                for r in results {
                    println!(
                        "{} distance={:.6} score={:.6}",
                        r.id.unwrap_or_default(),
                        r.distance,
                        r.score
                    );
                }
                println!("explain: {}", serde_json::to_string_pretty(&ex)?);
            } else {
                let results = col.query(&vector, top_k, None, 64)?;
                for r in results {
                    println!(
                        "{} distance={:.6} score={:.6}",
                        r.id.unwrap_or_default(),
                        r.distance,
                        r.score
                    );
                }
            }
        }
        Commands::Plan {
            size,
            dimension,
            top_k,
        } => {
            let plan = dv_query::IndexPlanner::plan(&dv_query::QueryPlannerInput {
                collection_size: size,
                dimension,
                top_k,
                has_filter: false,
            });
            println!("recommended_index: {:?}", plan.index_kind);
            println!("ef: {}", plan.ef);
            println!("reason: {}", plan.reason);
        }
        Commands::ShardCreate {
            name,
            shards,
            dimension,
            metric,
            index,
        } => {
            let metric = DistanceMetric::from_str(&metric)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e))?;
            let index_kind = match index.to_lowercase().as_str() {
                "flat" => IndexKind::Flat,
                "hnsw" => IndexKind::Hnsw,
                _ => IndexKind::ZColumn,
            };
            db.create_sharded_collection(&name, shards, dimension, metric, index_kind)?;
            println!(
                "created sharded collection '{name}' ({shards} fractal shards, dim={dimension}, index={index})"
            );
        }
        Commands::BatchSearch {
            collection,
            vectors_file,
            top_k,
            sharded,
        } => {
            let raw = std::fs::read_to_string(&vectors_file)?;
            let queries: Vec<Vec<f32>> = raw
                .lines()
                .filter(|l| !l.trim().is_empty())
                .map(|line| {
                    line.split(',')
                        .map(|s| s.trim().parse::<f32>())
                        .collect::<Result<Vec<_>, _>>()
                })
                .collect::<Result<Vec<_>, _>>()?;
            let refs: Vec<&[f32]> = queries.iter().map(|v| v.as_slice()).collect();
            if sharded {
                let batches = db.query_sharded_batch(&collection, &refs, top_k, None, 64)?;
                for (i, results) in batches.iter().enumerate() {
                    println!("query[{i}]:");
                    for r in results {
                        println!(
                            "  {} distance={:.6}",
                            r.id.as_deref().unwrap_or("?"),
                            r.distance
                        );
                    }
                }
            } else {
                let col = db.get_collection(&collection)?;
                let dim = col.config().dimension;
                for (i, q) in queries.iter().enumerate() {
                    if q.len() != dim {
                        return Err(format!("query {i} dim {} != {dim}", q.len()).into());
                    }
                }
                let batches = col.query_batch(&refs, top_k, None, 64)?;
                for (i, results) in batches.iter().enumerate() {
                    println!("query[{i}]:");
                    for r in results {
                        println!(
                            "  {} distance={:.6}",
                            r.id.as_deref().unwrap_or("?"),
                            r.distance
                        );
                    }
                }
            }
        }
        Commands::ShardServe { collection, bind } => {
            let server = ShardQueryServer::start(ShardServerConfig {
                data_dir: cli.data_dir.clone(),
                collection: collection.clone(),
                bind_addr: bind,
            })?;
            println!(
                "shard server listening on {} for collection '{collection}'",
                server.base_url()
            );
            loop {
                std::thread::park();
            }
        }
        Commands::Prove {
            million,
            scale,
            queries,
        } => {
            let mut config = ProveConfig {
                num_queries: queries,
                ..ProveConfig::default()
            };
            if let Some(s) = scale {
                config.scales = vec![s];
            } else if million {
                config.scales.push(1_000_000);
            }
            eprintln!(
                "running commercial proof at scales {:?} ({} queries each)...",
                config.scales, config.num_queries
            );
            let report = dv_bench::run(config);
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
    }

    Ok(())
}
