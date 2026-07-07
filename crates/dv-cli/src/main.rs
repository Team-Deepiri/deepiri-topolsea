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
            println!("vectors: {}", col.len());
        }
    }

    Ok(())
}
