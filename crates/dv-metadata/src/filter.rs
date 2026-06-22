use serde_json::Value;

#[derive(Debug, Clone, PartialEq)]
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
    And(Vec<Filter>),
    Or(Vec<Filter>),
}

impl Filter {
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
            }
            // Shorthand: { "field": value }
            if obj.len() == 1 {
                let (field, val) = obj.iter().next().unwrap();
                return Ok(Filter::Eq {
                    field: field.clone(),
                    value: val.clone(),
                });
            }
        }
        Err(dv_types::TopolseaError::Metadata(
            "unsupported filter expression".into(),
        ))
    }

    pub fn matches(&self, metadata: &Value) -> bool {
        match self {
            Filter::Eq { field, value } => metadata.get(field).map(|v| v == value).unwrap_or(false),
            Filter::And(items) => items.iter().all(|f| f.matches(metadata)),
            Filter::Or(items) => items.iter().any(|f| f.matches(metadata)),
        }
    }
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
}
