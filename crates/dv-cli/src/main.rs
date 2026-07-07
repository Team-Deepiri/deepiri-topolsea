use clap::{Parser, Subcommand};
use dv_query::Database;
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
    }

    Ok(())
}
