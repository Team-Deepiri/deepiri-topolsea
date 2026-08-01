use serde_json::Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterOp {
    Eq,
    Ne,
    Gt,
    Gte,
    Lt,
    Lte,
    In,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Filter {
    /// field must equal value
    Eq {
        field: String,
        value: Value,
    },
    /// Comparison / inequality / membership on a field.
    Cmp {
        field: String,
        op: FilterOp,
        value: Value,
    },
    And(Vec<Filter>),
    Or(Vec<Filter>),
}

impl Filter {
    /// Parse JSON filter dialect.
    ///
    /// Supported forms:
    /// - Shorthand equality: `{"tag": "alpha"}`
    /// - Operators: `{"score": {"$gt": 1}}`, `{"tag": {"$ne": "x"}}`, `{"tag": {"$in": ["a","b"]}}`
    /// - Combinators: `{"$and": [...]}`, `{"$or": [...]}`
    pub fn from_json(value: &Value) -> dv_types::Result<Self> {
        if let Some(obj) = value.as_object() {
            if obj.len() == 1 {
                if let Some(and) = obj.get("$and") {
                    let items = and.as_array().ok_or_else(|| {
                        dv_types::TopolseaError::Metadata("$and must be array".into())
                    })?;
                    return Ok(Filter::And(
                        items
                            .iter()
                            .map(Filter::from_json)
                            .collect::<dv_types::Result<_>>()?,
                    ));
                }
                if let Some(or) = obj.get("$or") {
                    let items = or.as_array().ok_or_else(|| {
                        dv_types::TopolseaError::Metadata("$or must be array".into())
                    })?;
                    return Ok(Filter::Or(
                        items
                            .iter()
                            .map(Filter::from_json)
                            .collect::<dv_types::Result<_>>()?,
                    ));
                }

                let (field, val) = obj.iter().next().unwrap();
                if let Some(op_obj) = val.as_object() {
                    return parse_field_ops(field, op_obj);
                }
                return Ok(Filter::Eq {
                    field: field.clone(),
                    value: val.clone(),
                });
            }

            // Multi-field object ⇒ implicit $and of each field clause.
            if !obj.is_empty() {
                let mut parts = Vec::with_capacity(obj.len());
                for (field, val) in obj {
                    if field.starts_with('$') {
                        return Err(dv_types::TopolseaError::Metadata(format!(
                            "unexpected top-level operator {field} in multi-key filter"
                        )));
                    }
                    if let Some(op_obj) = val.as_object() {
                        parts.push(parse_field_ops(field, op_obj)?);
                    } else {
                        parts.push(Filter::Eq {
                            field: field.clone(),
                            value: val.clone(),
                        });
                    }
                }
                return Ok(Filter::And(parts));
            }
        }
        Err(dv_types::TopolseaError::Metadata(
            "unsupported filter expression".into(),
        ))
    }

    /// Serialize back to the JSON filter dialect (for remote shard fan-out).
    pub fn to_json(&self) -> Value {
        match self {
            Filter::Eq { field, value } => {
                let mut map = serde_json::Map::new();
                map.insert(field.clone(), value.clone());
                Value::Object(map)
            }
            Filter::Cmp { field, op, value } => {
                let op_key = match op {
                    FilterOp::Eq => "$eq",
                    FilterOp::Ne => "$ne",
                    FilterOp::Gt => "$gt",
                    FilterOp::Gte => "$gte",
                    FilterOp::Lt => "$lt",
                    FilterOp::Lte => "$lte",
                    FilterOp::In => "$in",
                };
                let mut inner = serde_json::Map::new();
                inner.insert(op_key.to_string(), value.clone());
                let mut map = serde_json::Map::new();
                map.insert(field.clone(), Value::Object(inner));
                Value::Object(map)
            }
            Filter::And(parts) => Value::Object(serde_json::Map::from_iter([(
                "$and".into(),
                Value::Array(parts.iter().map(|p| p.to_json()).collect()),
            )])),
            Filter::Or(parts) => Value::Object(serde_json::Map::from_iter([(
                "$or".into(),
                Value::Array(parts.iter().map(|p| p.to_json()).collect()),
            )])),
        }
    }

    pub fn matches(&self, metadata: &Value) -> bool {
        match self {
            Filter::Eq { field, value } => metadata.get(field).map(|v| v == value).unwrap_or(false),
            Filter::Cmp { field, op, value } => {
                let actual = metadata.get(field);
                match op {
                    FilterOp::Eq => actual.map(|v| v == value).unwrap_or(false),
                    FilterOp::Ne => actual.map(|v| v != value).unwrap_or(true),
                    FilterOp::In => match value.as_array() {
                        Some(arr) => actual.map(|v| arr.iter().any(|x| x == v)).unwrap_or(false),
                        None => false,
                    },
                    FilterOp::Gt | FilterOp::Gte | FilterOp::Lt | FilterOp::Lte => {
                        let Some(actual) = actual else {
                            return false;
                        };
                        compare_ordered(actual, value, *op)
                    }
                }
            }
            Filter::And(items) => items.iter().all(|f| f.matches(metadata)),
            Filter::Or(items) => items.iter().any(|f| f.matches(metadata)),
        }
    }
}

fn parse_field_ops(
    field: &str,
    op_obj: &serde_json::Map<String, Value>,
) -> dv_types::Result<Filter> {
    if op_obj.len() == 1 {
        let (op_key, op_val) = op_obj.iter().next().unwrap();
        let op = match op_key.as_str() {
            "$eq" => FilterOp::Eq,
            "$ne" => FilterOp::Ne,
            "$gt" => FilterOp::Gt,
            "$gte" => FilterOp::Gte,
            "$lt" => FilterOp::Lt,
            "$lte" => FilterOp::Lte,
            "$in" => {
                if !op_val.is_array() {
                    return Err(dv_types::TopolseaError::Metadata(
                        "$in value must be an array".into(),
                    ));
                }
                FilterOp::In
            }
            other => {
                return Err(dv_types::TopolseaError::Metadata(format!(
                    "unsupported filter operator {other}"
                )));
            }
        };
        if op == FilterOp::Eq {
            return Ok(Filter::Eq {
                field: field.to_string(),
                value: op_val.clone(),
            });
        }
        return Ok(Filter::Cmp {
            field: field.to_string(),
            op,
            value: op_val.clone(),
        });
    }

    // Multiple ops on one field ⇒ AND.
    let mut parts = Vec::new();
    for (op_key, op_val) in op_obj {
        let single = serde_json::json!({ field: { op_key: op_val } });
        parts.push(Filter::from_json(&single)?);
    }
    Ok(Filter::And(parts))
}

fn compare_ordered(actual: &Value, expected: &Value, op: FilterOp) -> bool {
    match (as_f64(actual), as_f64(expected)) {
        (Some(a), Some(b)) => match op {
            FilterOp::Gt => a > b,
            FilterOp::Gte => a >= b,
            FilterOp::Lt => a < b,
            FilterOp::Lte => a <= b,
            _ => false,
        },
        _ => match (actual.as_str(), expected.as_str()) {
            (Some(a), Some(b)) => match op {
                FilterOp::Gt => a > b,
                FilterOp::Gte => a >= b,
                FilterOp::Lt => a < b,
                FilterOp::Lte => a <= b,
                _ => false,
            },
            _ => false,
        },
    }
}

fn as_f64(v: &Value) -> Option<f64> {
    v.as_f64()
        .or_else(|| v.as_i64().map(|i| i as f64))
        .or_else(|| v.as_u64().map(|u| u as f64))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn eq_filter() {
        let f = Filter::from_json(&json!({"topic": "rust"})).unwrap();
        assert!(f.matches(&json!({"topic": "rust", "n": 1})));
        assert!(!f.matches(&json!({"topic": "python"})));
    }

    #[test]
    fn ne_gt_in_filters() {
        let ne = Filter::from_json(&json!({"tag": {"$ne": "alpha"}})).unwrap();
        assert!(ne.matches(&json!({"tag": "beta"})));
        assert!(!ne.matches(&json!({"tag": "alpha"})));
        assert!(ne.matches(&json!({})));

        let gt = Filter::from_json(&json!({"n": {"$gt": 5}})).unwrap();
        assert!(gt.matches(&json!({"n": 6})));
        assert!(!gt.matches(&json!({"n": 5})));

        let gte = Filter::from_json(&json!({"n": {"$gte": 5}})).unwrap();
        assert!(gte.matches(&json!({"n": 5})));

        let lt = Filter::from_json(&json!({"n": {"$lt": 5}})).unwrap();
        assert!(lt.matches(&json!({"n": 4})));

        let lte = Filter::from_json(&json!({"n": {"$lte": 5}})).unwrap();
        assert!(lte.matches(&json!({"n": 5})));

        let inn = Filter::from_json(&json!({"tag": {"$in": ["a", "b"]}})).unwrap();
        assert!(inn.matches(&json!({"tag": "a"})));
        assert!(!inn.matches(&json!({"tag": "c"})));
    }

    #[test]
    fn and_or_with_ops() {
        let f = Filter::from_json(&json!({
            "$and": [
                {"tag": {"$ne": "x"}},
                {"n": {"$gte": 1}}
            ]
        }))
        .unwrap();
        assert!(f.matches(&json!({"tag": "y", "n": 2})));
        assert!(!f.matches(&json!({"tag": "x", "n": 2})));
    }
}
