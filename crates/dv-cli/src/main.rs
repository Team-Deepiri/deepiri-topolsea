use clap::{Parser, Subcommand};
use dv_query::Database;
use dv_types::DistanceMetric;
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
        } => {
            let metric = DistanceMetric::from_str(&metric)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e))?;
            db.create_collection(dv_types::CollectionConfig::new(
                name.clone(),
                dimension,
                metric,
            ))?;
            println!("created collection '{name}' (dim={dimension}, metric={metric})");
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
