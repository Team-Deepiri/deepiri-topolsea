use dv_types::{IndexKind, ZColumnConfig};

/// Workload hints for index selection and tuning.
#[derive(Debug, Clone)]
pub struct QueryPlannerInput {
    pub collection_size: usize,
    pub dimension: usize,
    pub top_k: usize,
    pub has_filter: bool,
}

/// Recommended execution plan for a vector query.
#[derive(Debug, Clone)]
pub struct QueryPlan {
    pub index_kind: IndexKind,
    pub ef: usize,
    pub fetch_multiplier: usize,
    pub reason: String,
}

/// Heuristic planner — picks Z-Column for large, filter-light ANN workloads.
pub struct IndexPlanner;

impl IndexPlanner {
    pub fn plan(input: &QueryPlannerInput) -> QueryPlan {
        if input.collection_size < 500 {
            return QueryPlan {
                index_kind: IndexKind::Flat,
                ef: 0,
                fetch_multiplier: 1,
                reason: "small collection — exact flat search".into(),
            };
        }

        if input.has_filter && input.collection_size < 5_000 {
            return QueryPlan {
                index_kind: IndexKind::Flat,
                ef: 0,
                fetch_multiplier: 10,
                reason: "selective filter on medium collection".into(),
            };
        }

        if input.dimension >= 64 && input.collection_size >= 1_000 {
            let ef = ZColumnConfig::default().ef_search.max(input.top_k * 4);
            return QueryPlan {
                index_kind: IndexKind::ZColumn,
                ef,
                fetch_multiplier: ZColumnConfig::default().hybrid_rerank_pool,
                reason: "high-D large collection — fractal Z-Column with hybrid rerank".into(),
            };
        }

        QueryPlan {
            index_kind: IndexKind::Hnsw,
            ef: 64.max(input.top_k * 4),
            fetch_multiplier: 1,
            reason: "default ANN — HNSW graph traversal".into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recommends_zcolumn_at_scale() {
        let plan = IndexPlanner::plan(&QueryPlannerInput {
            collection_size: 10_000,
            dimension: 128,
            top_k: 10,
            has_filter: false,
        });
        assert_eq!(plan.index_kind, IndexKind::ZColumn);
    }
}
